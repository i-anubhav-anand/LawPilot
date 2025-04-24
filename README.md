# Legal AI Assistant Backend

This is the backend for the Legal AI Assistant, a Retrieval-Augmented Generation (RAG) system specializing in California tenant-landlord law. The system uses a combination of legal corpus data, user documents, and structured conversation to provide accurate legal information.

## Features

- **RAG-Based Legal Assistant**: Provides accurate legal information with citations to relevant legal sources.
- **Document Upload and Analysis**: Process uploaded documents (leases, notices, etc.) and analyze them for legal issues.
- **Case File Management**: Maintain structured YAML case files with facts gathered during conversations.
- **Conversation History**: Track and maintain conversational context for better responses.

## Technology Stack

- **FastAPI**: Web framework for building the API
- **LangChain**: For building RAG applications
- **OpenAI API**: For embeddings and text generation
- **FAISS**: Vector database for efficient similarity search
- **SentenceTransformers**: For generating embeddings locally
- **PyPDF, docx2txt, pytesseract**: For document parsing

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository
2. Navigate to the backend directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the backend directory with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

### Running the Server

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Chat Endpoints

- `POST /api/chat/`: Send a message to the legal assistant
- `GET /api/chat/history/{session_id}`: Get chat history for a session

### Document Endpoints

- `POST /api/documents/upload`: Upload a document for analysis
- `GET /api/documents/{document_id}`: Get status of a document
- `POST /api/documents/{document_id}/analyze`: Analyze an uploaded document

### Case File Endpoints

- `POST /api/case-files/`: Create a new case file
- `GET /api/case-files/{case_file_id}`: Get a case file
- `PUT /api/case-files/{case_file_id}`: Update a case file with new facts
- `GET /api/case-files/`: List all case files
- `DELETE /api/case-files/{case_file_id}`: Delete a case file

## Adding Legal Corpus Data

To add legal corpus data:

1. Place legal text files (.txt) in a directory
2. Use the corpus loader utility to index them:

```python
from app.utils.corpus_loader import CorpusLoader

loader = CorpusLoader()
await loader.load_corpus_directory("path/to/corpus", corpus_type="law")
```

## Project Structure

- `app/api/`: API endpoints
- `app/core/`: Core functionality (RAG engine, document processing, etc.)
- `app/models/`: Pydantic models
- `app/utils/`: Utility functions
- `legal_corpus/`: Legal corpus data
- `uploads/`: User uploaded documents
- `case_files/`: YAML case files
- `chat_histories/`: Chat history data

## License

MIT 