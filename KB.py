import os
import logging
import time
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import PyPDF2
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client.http import models

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "Medical_Coder"
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KB")

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # A good general-purpose model

class SentenceTransformerEmbeddings:
    """Wrapper for sentence_transformers embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the sentence transformer model"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the sentence transformer model"""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

def preprocess_text(text: str) -> str:
    """Preprocess the extracted text"""
    text = " ".join(text.split())  # Basic whitespace normalization
    return text

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract text from a PDF file"""
    try:
        text = ""
        metadata = {
            "source": pdf_path,
            "file_type": "pdf",
            "file_name": os.path.basename(pdf_path)
        }
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
        return text.strip(), metadata
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return None, None

def extract_text_from_excel(excel_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract text from an Excel file, processing each sheet separately"""
    try:
        results = []
        excel_file = pd.ExcelFile(excel_path)
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Convert DataFrame to text
            text = ""
            for _, row in df.iterrows():
                text += " ".join(str(cell) for cell in row if pd.notna(cell)) + "\n"
            
            metadata = {
                "source": excel_path,
                "file_type": "excel",
                "file_name": os.path.basename(excel_path),
                "sheet_name": sheet_name
            }
            
            results.append((text.strip(), metadata))
            
        return results
    except Exception as e:
        logger.error(f"Error extracting text from Excel {excel_path}: {e}")
        return []

def create_chunks(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """Split text into chunks with metadata"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_text(text)
    documents = []
    for i, chunk in enumerate(chunks):
        processed_text = preprocess_text(chunk)
        doc_metadata = metadata.copy()
        doc_metadata["chunk_size"] = len(processed_text)
        doc_metadata["chunk_index"] = i
        documents.append(Document(
            page_content=processed_text,
            metadata=doc_metadata
        ))
    return documents

def is_document_processed(qdrant_client: QdrantClient, file_path: str) -> bool:
    """Check if a document with the given file path is already stored in Qdrant"""
    try:
        # Search Qdrant for documents with the same file path
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={"must": [{"key": "metadata.source", "match": {"value": file_path}}]},
            limit=1,
        )
        return len(results[0]) > 0
    except Exception as e:
        logger.error(f"Error checking if document is processed: {e}")
        return False

def store_in_qdrant(qdrant_client: QdrantClient, embeddings: SentenceTransformerEmbeddings, 
                   documents: List[Document], max_retries: int = 3) -> bool:
    """Store processed documents in Qdrant DB with retry logic"""
    for attempt in range(max_retries):
        try:
            # Create vectors directly instead of using from_documents to avoid pickling issues
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Generate embeddings
            embeddings_list = embeddings.embed_documents(texts)
            
            # Process in smaller batches to avoid timeout
            batch_size = 50  # Smaller batch size to avoid timeout
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            from tqdm import tqdm
            for i in tqdm(range(0, len(documents), batch_size), desc="Batches", total=total_batches):
                # Create points with IDs, vectors, and payloads for this batch
                batch_points = []
                end_idx = min(i + batch_size, len(documents))
                
                for j, (text, metadata, embedding) in enumerate(zip(
                    texts[i:end_idx], 
                    metadatas[i:end_idx], 
                    embeddings_list[i:end_idx]
                )):
                    batch_points.append({
                        "id": i + j,  # Ensure unique IDs across batches
                        "vector": embedding,
                        "payload": {"text": text, "metadata": metadata}
                    })
                
                # Upsert batch points directly using the client
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch_points,
                    wait=True  # Ensure the operation completes before moving to next batch
                )
            
            logger.info(f"Stored {len(documents)} document chunks in Qdrant")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)  # Wait before retrying
    logger.error(f"Failed to store documents after {max_retries} attempts")
    return False

def process_documents(kb_dir: str = KB_DIR) -> str:
    """Main processing pipeline for PDF and Excel files"""
    all_documents = []
    skipped_files = 0
    processed_files = 0
    
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=120,  # Increase timeout to 120 seconds
        )
        try:
            qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.source",
            field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info("Created index on 'metadata.source'")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("Index on 'metadata.source' already exists")
            else:
                logger.warning(f"Could not create index on 'metadata.source': {e}")

        # Initialize embeddings
        embeddings = SentenceTransformerEmbeddings()

        # Process PDF files
        pdf_files = glob.glob(os.path.join(kb_dir, "*.pdf"))
        for pdf_file in pdf_files:
            logger.info(f"Processing PDF file: {pdf_file}")
            
            # Skip if document is already processed
            if is_document_processed(qdrant_client, pdf_file):
                logger.info(f"Skipping already processed file: {pdf_file}")
                skipped_files += 1
                continue
                
            text, metadata = extract_text_from_pdf(pdf_file)
            if not text or not metadata:
                skipped_files += 1
                continue
                
            # Create chunks and add to documents
            chunks = create_chunks(text, metadata)
            all_documents.extend(chunks)
            processed_files += 1
            
        # Process Excel files
        excel_files = glob.glob(os.path.join(kb_dir, "*.xlsx")) + glob.glob(os.path.join(kb_dir, "*.xls"))
        for excel_file in excel_files:
            logger.info(f"Processing Excel file: {excel_file}")
            
            # Skip if document is already processed
            if is_document_processed(qdrant_client, excel_file):
                logger.info(f"Skipping already processed file: {excel_file}")
                skipped_files += 1
                continue
                
            sheet_results = extract_text_from_excel(excel_file)
            if not sheet_results:
                skipped_files += 1
                continue
                
            for text, metadata in sheet_results:
                # Create chunks and add to documents
                chunks = create_chunks(text, metadata)
                all_documents.extend(chunks)
            processed_files += 1
    
        if all_documents:
            success = store_in_qdrant(qdrant_client, embeddings, all_documents)
            if success:
                return f"Successfully processed {processed_files} files ({len(all_documents)} chunks). Skipped {skipped_files} files."
            else:
                return "Failed to store documents in Qdrant."
        return "No valid documents processed."
    
    except Exception as e:
        logger.error(f"Critical error processing documents: {e}")
        return "Processing failed."

if __name__ == '__main__':
    result = process_documents()
    print(result)