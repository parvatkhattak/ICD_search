import os
import logging
import json
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from functools import lru_cache
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store indexed data
icd10_data = {}
icd10_additional_data = {}
code_index = {}
prefix_index = {}  # New index for efficient prefix searching

# Router for search endpoints
router = APIRouter()

def load_data():
    """Load and index ICD-10 data for fast searching"""
    global icd10_data, icd10_additional_data, code_index, prefix_index
    
    start_time = time.time()
    try:
        # Load the main ICD-10 index data
        icd10_index_path = os.getenv("ICD10_INDEX_PATH")
        with open(icd10_index_path, 'r', encoding='utf-8') as f:
            icd10_data = json.load(f)
        
        # Load the additional ICD-10 data
        icd10_updated_path = os.getenv("ICD10_UPDATED_PATH")
        try:
            with open(icd10_updated_path, 'r', encoding='utf-8') as f:
                icd10_additional_data = json.load(f)
                
                # Create an index for fast code lookup
                if 'icd_codes' in icd10_additional_data:
                    # First pass: create exact match index
                    for i, entry in enumerate(icd10_additional_data['icd_codes']):
                        if isinstance(entry, dict) and 'lookupCode' in entry:
                            code = entry['lookupCode']
                            if isinstance(code, str):
                                normalized_code = code.replace('.', '').replace(' ', '').upper()
                                if normalized_code not in code_index:
                                    code_index[normalized_code] = []
                                code_index[normalized_code].append(i)
                                
                                # Create prefix index for first letter and first two letters
                                # This is more efficient than indexing all possible prefixes
                                if len(normalized_code) > 0:
                                    first_char = normalized_code[0]
                                    if first_char not in prefix_index:
                                        prefix_index[first_char] = []
                                    prefix_index[first_char].append(i)
                                    
                                    if len(normalized_code) > 1:
                                        first_two_chars = normalized_code[:2]
                                        if first_two_chars not in prefix_index:
                                            prefix_index[first_two_chars] = []
                                        prefix_index[first_two_chars].append(i)
                                        
                                        if len(normalized_code) > 2:
                                            first_three_chars = normalized_code[:3]
                                            if first_three_chars not in prefix_index:
                                                prefix_index[first_three_chars] = []
                                            prefix_index[first_three_chars].append(i)
                
            logger.info(f"Indexed {len(code_index)} exact ICD-10 codes and {len(prefix_index)} prefixes in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error loading ICD-10 additional data: {e}")
            icd10_additional_data = {}
    except Exception as e:
        logger.error(f"Error loading ICD-10 data: {e}")
        icd10_data = {}

# Use a larger cache size for better performance with common searches
@lru_cache(maxsize=512)
def _search_code_cached(normalized_code: str, limit: int = 20):
    """Cached implementation of code search logic with optimized prefix matching"""
    start_time = time.time()
    matching_entries = []
    exact_matches = set()
    
    # First look for exact matches using the index (fastest path)
    if normalized_code in code_index:
        for idx in code_index[normalized_code]:
            matching_entries.append(icd10_additional_data['icd_codes'][idx])
            exact_matches.add(idx)
    
    # If we don't have enough results, look for partial matches using prefix index
    if len(matching_entries) < limit:
        # First try to find matches using the prefix index (much faster)
        candidate_indices = set()
        max_candidates = 500  # Limit the number of candidates to process
        
        # Use the most specific prefix available for better performance
        best_prefix = ""
        if len(normalized_code) >= 1 and normalized_code[0] in prefix_index:
            best_prefix = normalized_code[0]
            if len(normalized_code) >= 2 and normalized_code[:2] in prefix_index:
                best_prefix = normalized_code[:2]
                if len(normalized_code) >= 3 and normalized_code[:3] in prefix_index:
                    best_prefix = normalized_code[:3]
        
        # If we found a good prefix, use it to get candidates
        if best_prefix:
            candidate_indices.update(prefix_index[best_prefix][:max_candidates])
            
            # For 'S' codes (which are common injury codes), optimize further
            if normalized_code.startswith('S') and len(normalized_code) >= 2:
                # Add specific candidates that start with the same two characters
                for prefix in prefix_index:
                    if prefix.startswith(normalized_code[:2]) and len(prefix) > len(best_prefix):
                        candidate_indices.update(prefix_index[prefix][:100])  # Limit per prefix
        
        # Filter candidates that actually match our search criteria
        partial_matches = []
        processed = 0
        
        # Process candidates more efficiently
        for idx in candidate_indices:
            processed += 1
            if processed > max_candidates:
                break  # Limit processing to avoid timeouts
                
            if idx not in exact_matches:
                entry = icd10_additional_data['icd_codes'][idx]
                if 'lookupCode' in entry:
                    code = entry['lookupCode']
                    if isinstance(code, str):
                        entry_code = code.replace('.', '').replace(' ', '').upper()
                        
                        # Fast check - if entry code starts with our search code, it's a good match
                        if entry_code.startswith(normalized_code):
                            # Perfect prefix match gets high score
                            partial_matches.append((len(normalized_code), idx))
                        elif normalized_code.startswith(entry_code):
                            # Search code starts with entry code
                            partial_matches.append((len(entry_code), idx))
                        # Only do more expensive character-by-character comparison if needed
                        elif best_prefix and entry_code.startswith(best_prefix):
                            # Calculate match score based on length of common prefix
                            common_len = len(best_prefix)
                            for i in range(len(best_prefix), min(len(normalized_code), len(entry_code))):
                                if normalized_code[i] == entry_code[i]:
                                    common_len += 1
                                else:
                                    break
                            if common_len > len(best_prefix):  # Only add if better than the prefix match
                                partial_matches.append((common_len, idx))
        
        # Sort partial matches by relevance (longer common prefix = better match)
        partial_matches.sort(key=lambda x: -x[0])
        
        # Add partial matches up to the limit
        for _, idx in partial_matches[:limit - len(matching_entries)]:
            matching_entries.append(icd10_additional_data['icd_codes'][idx])
    
    # Limit results to prevent overwhelming the UI
    matching_entries = matching_entries[:limit]
    
    logger.info(f"Search for '{normalized_code}' found {len(matching_entries)} results in {time.time() - start_time:.4f} seconds (from {len(exact_matches)} exact, {len(matching_entries) - len(exact_matches)} partial)")
    return matching_entries

@router.get("/search/{code}")
async def search_code(code: str, limit: int = Query(20, ge=1, le=100)):
    """API endpoint to quickly search for ICD-10 codes"""
    try:
        # Normalize the code (remove dots, spaces, convert to uppercase)
        normalized_code = code.replace('.', '').replace(' ', '').upper()
        
        # Use the cached search function
        matching_entries = _search_code_cached(normalized_code, limit)
        
        return JSONResponse(content=matching_entries)
    except Exception as e:
        logger.error(f"Error searching code {code}: {e}")
        return JSONResponse(content={"error": f"Failed to search code: {str(e)}"}, status_code=500)

# Add a route to force reload data (useful for development/testing)
@router.get("/reload-icd-data")
async def reload_data():
    """Force reload of ICD-10 data (admin only)"""
    try:
        start_time = time.time()
        load_data()
        duration = time.time() - start_time
        return JSONResponse(content={
            "status": "success", 
            "message": f"Data reloaded successfully in {duration:.2f} seconds",
            "stats": {
                "exact_codes": len(code_index),
                "prefix_indices": len(prefix_index)
            }
        })
    except Exception as e:
        logger.error(f"Error reloading data: {e}")
        return JSONResponse(content={"error": f"Failed to reload data: {str(e)}"}, status_code=500)

# Initialize data when module is imported
load_data()

# Create FastAPI app and include router
def create_app():
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="ICD-10 Search API",
        description="Fast search API for ICD-10 medical codes",
        version="1.0.0"
    )
    
    # Add CORS middleware - THIS IS THE KEY FIX
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins like ["http://localhost:3000", "https://yourdomain.com"]
        allow_credentials=True,
        allow_methods=["*"],  # Or specify specific methods like ["GET", "POST"]
        allow_headers=["*"],  # Or specify specific headers
    )
    
    # Include the router
    app.include_router(router, prefix="/api/v1", tags=["ICD-10 Search"])
    
    # Add a simple health check endpoint
    @app.get("/")
    async def root():
        return {"message": "ICD-10 Search API is running", "status": "healthy"}
    
    return app

# Main entry point for running the application
if __name__ == "__main__":
    print("Starting ICD-10 Search API server...")
    
    # Create the FastAPI app
    app = create_app()
    
    # Configure server settings
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    print(f"Server will run on http://{host}:{port}")
    print("Available endpoints:")
    print(f"  - Health check: http://{host}:{port}/")
    print(f"  - Search codes: http://{host}:{port}/api/v1/search/{{code}}")
    print(f"  - Reload data: http://{host}:{port}/api/v1/reload-icd-data")
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )