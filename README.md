# AI PDF Chat API

A Flask-based REST API for uploading PDF documents and having intelligent conversations with their content using LangChain and OpenAI.

## Features

- 📄 PDF file upload and text extraction using LangChain
- 🤖 AI-powered chat using OpenAI GPT models
- 🔍 Vector-based document search with FAISS
- 💬 Chat history management
- 🌐 CORS enabled for frontend integration
- 📊 Document management (list, delete)
- ☁️ Automatic file hosting via tmpfiles.org

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up OpenAI API key:**
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key-here
```

3. **Run the Flask application:**
```bash
python run.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/health` - Check API status and OpenAI configuration

### Document Management
- **POST** `/upload` - Upload a PDF file
- **GET** `/documents` - List all uploaded documents
- **DELETE** `/documents/<document_id>` - Delete a document

### Chat
- **POST** `/chat/<document_id>` - Send a message to chat with document
- **GET** `/chat/<document_id>/history` - Get chat history for a document

## Usage Examples

### Upload a PDF
```bash
curl -X POST -F "file=@document.pdf" http://localhost:5000/upload
```

### Chat with document
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"message": "What is this document about?"}' \
  http://localhost:5000/chat/<document_id>
```

### Get chat history
```bash
curl http://localhost:5000/chat/<document_id>/history
```

## How It Works

1. **PDF Upload**: Files are uploaded and automatically hosted on tmpfiles.org for public access
2. **Text Extraction**: LangChain's PyPDFLoader extracts text from the PDF
3. **Text Chunking**: Documents are split into manageable chunks using RecursiveCharacterTextSplitter
4. **Vector Storage**: Text chunks are embedded using OpenAI embeddings and stored in FAISS
5. **AI Chat**: User questions are processed using LangChain's RetrievalQA chain with OpenAI GPT models

## Response Format

All API responses follow this format:
```json
{
  "success": true,
  "data": "...",
  "error": "..." // only present if success is false
}
```

## Configuration

Environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `FLASK_ENV` - Flask environment (default: development)
- `FLASK_DEBUG` - Enable debug mode (default: True)

## Limitations

- Files are temporarily stored and processed in memory
- Maximum file size: 16MB
- Only PDF files are supported
- Documents are hosted publicly on tmpfiles.org (temporary)

## Production Considerations

For production deployment:
- Implement proper database storage instead of in-memory storage
- Add authentication and rate limiting
- Use secure file storage (AWS S3, etc.) instead of tmpfiles.org
- Add input validation and security measures
- Implement proper logging and monitoring
- Consider using a production WSGI server (Gunicorn, uWSGI)

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your API key is set correctly
   - Check that you have sufficient OpenAI credits

2. **PDF Processing Error**
   - Ensure the PDF is not corrupted
   - Check that the PDF contains extractable text (not just images)

3. **Memory Issues**
   - Large PDFs may cause memory issues
   - Consider implementing file size limits

4. **Network Issues**
   - Ensure tmpfiles.org is accessible
   - Check firewall settings for outbound connections