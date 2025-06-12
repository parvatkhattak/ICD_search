# ICD-10 Search API

A high-performance FastAPI-based search system for ICD-10 medical codes with vector database integration using Qdrant. This project enables fast, intelligent searching of medical codes and maintains a knowledge base of medical documents.

## Features

- **Fast ICD-10 Code Search**: Optimized prefix-based search with caching for sub-second response times
- **Vector Database Integration**: Qdrant-powered knowledge base for medical documents
- **Multi-format Document Processing**: Support for PDF and Excel files
- **Hierarchical Data Structure**: Converts CSV data to hierarchical JSON for better organization
- **RESTful API**: Clean, documented API endpoints with CORS support
- **Intelligent Indexing**: Multiple indexing strategies for exact and partial matches

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Qdrant DB     │    │   File System  │
│                 │    │                 │    │                 │
│ • Search API    │◄──►│ • Vector Store  │◄──►│ • PDFs         │
│ • CORS Support  │    │ • Embeddings    │    │ • Excel Files  │
│ • Caching       │    │ • Collections   │    │ • JSON Data    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

- Python 3.8+
- Qdrant server (cloud or local)
- Required Python packages (see requirements below)

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd icd10-search
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn qdrant-client sentence-transformers
   pip install langchain langchain-qdrant PyPDF2 pandas python-dotenv
   pip install python-multipart tqdm
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ICD10_INDEX_PATH=path/to/icd10_index.json
   ICD10_UPDATED_PATH=path/to/icd10_additional_data.json
   HOST=127.0.0.1
   PORT=8000
   DEBUG=False
   ```

4. **Create the KB directory**
   ```bash
   mkdir KB
   ```
   Place your PDF and Excel files in this directory.

## Quick Start

### 1. Initialize Qdrant Collection
```bash
python create_collection.py
```

### 2. Convert CSV to JSON (if needed)
```bash
python csv_to_json_converter.py
```

### 3. Process Documents into Knowledge Base
```bash
python KB.py
```

### 4. Start the API Server
```bash
python icd_search.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```http
GET /
```
Returns server status and health information.

### Search ICD-10 Codes
```http
GET /api/v1/search/{code}?limit=20
```

**Parameters:**
- `code` (required): ICD-10 code to search for
- `limit` (optional): Maximum number of results (1-100, default: 20)

**Example:**
```bash
curl "http://localhost:8000/api/v1/search/S72?limit=10"
```

### Reload Data
```http
GET /api/v1/reload-icd-data
```
Force reload of ICD-10 data (useful for development).

## File Structure

```
project/
├── icd_search.py              # Main FastAPI application
├── create_collection.py       # Qdrant collection setup
├── csv_to_json_converter.py   # CSV to hierarchical JSON converter
├── KB.py                      # Knowledge base document processor
├── KB/                        # Knowledge base directory
│   ├── *.pdf                  # PDF documents
│   ├── *.xlsx                 # Excel files
│   └── *.json                 # JSON data files
├── .env                       # Environment variables
└── README.md                  # This file
```

## Key Components

### ICD Search Engine (`icd_search.py`)
- **Optimized Indexing**: Creates exact match and prefix indices for fast lookups
- **Caching**: LRU cache for frequently searched codes
- **Smart Partial Matching**: Intelligent prefix matching with relevance scoring
- **Performance Monitoring**: Request timing and result statistics

### Knowledge Base Processor (`KB.py`)
- **Multi-format Support**: Handles PDF and Excel files
- **Text Chunking**: Intelligent text splitting with overlap
- **Vector Embeddings**: Uses SentenceTransformer for semantic search
- **Batch Processing**: Efficient batch uploads to Qdrant
- **Duplicate Detection**: Prevents reprocessing of existing documents

### Data Converter (`csv_to_json_converter.py`)
- **Hierarchical Structure**: Converts flat CSV to nested JSON
- **Level-based Organization**: Automatically detects and uses level information
- **Flexible Input**: Handles various CSV formats and encodings
- **Clean Output**: Removes empty children and optimizes structure

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | Required |
| `QDRANT_API_KEY` | Qdrant API key | Required |
| `ICD10_INDEX_PATH` | Path to main ICD-10 index file | Required |
| `ICD10_UPDATED_PATH` | Path to additional ICD-10 data | Required |
| `HOST` | Server host | 127.0.0.1 |
| `PORT` | Server port | 8000 |
| `DEBUG` | Debug mode | False |

### Performance Tuning

- **Batch Size**: Adjust `batch_size` in `KB.py` for document processing
- **Cache Size**: Modify `maxsize` in `@lru_cache` decorator
- **Timeout**: Increase Qdrant client timeout for large documents
- **Chunk Size**: Tune `chunk_size` and `chunk_overlap` in text splitter

## Usage Examples

### Search for Fracture Codes
```python
import requests

response = requests.get("http://localhost:8000/api/v1/search/S72")
results = response.json()

for result in results:
    print(f"Code: {result['lookupCode']} - {result.get('description', 'N/A')}")
```

### Add New Documents
1. Place PDF or Excel files in the `KB/` directory
2. Run `python KB.py` to process new documents
3. Documents are automatically indexed and searchable

## Development

### Running in Development Mode
```bash
# Set DEBUG=True in .env file
uvicorn icd_search:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Test the health endpoint
curl http://localhost:8000/

# Test search functionality
curl "http://localhost:8000/api/v1/search/S72?limit=5"

# Test data reload
curl http://localhost:8000/api/v1/reload-icd-data
```

## Deployment

### Production Deployment
1. Set `DEBUG=False` in environment variables
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn icd_search:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "icd_search:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Performance Optimization

- **Indexing Strategy**: Uses multi-level prefix indexing for O(1) lookups
- **Caching**: LRU cache prevents repeated processing of common searches
- **Batch Processing**: Efficient bulk operations with Qdrant
- **Memory Management**: Controlled candidate processing to prevent timeouts

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   - Verify `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
   - Check network connectivity to Qdrant server

2. **Out of Memory During Document Processing**
   - Reduce `batch_size` in `KB.py`
   - Process documents in smaller batches

3. **Slow Search Performance**
   - Check if data is properly indexed (run reload endpoint)
   - Verify cache is working (check logs for cache hits)

4. **CORS Issues**
   - Update `allow_origins` in CORS middleware
   - Ensure proper headers are being sent

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue in the repository
- Check the logs for detailed error messages
- Verify environment variables and file paths
