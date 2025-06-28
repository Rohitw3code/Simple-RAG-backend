from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import uuid
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
import shutil

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for demo purposes
documents = {}
chat_histories = {}
vectorstores = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_tmpfiles(file_path):
    """Upload file to tmpfiles.org and return public direct download URL"""
    try:
        upload_url = "https://tmpfiles.org/api/v1/upload"
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(upload_url, files=files)

        if response.status_code == 200:
            response_json = response.json()
            # Construct the direct download link
            url = response_json['data']['url']
            return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        else:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")
    except Exception as e:
        raise Exception(f"Error uploading to tmpfiles: {str(e)}")

def process_pdf_with_langchain(pdf_url, document_id):
    """Process PDF using LangChain and create vector store"""
    temp_pdf_path = None
    try:
        # Create a temporary file to store the downloaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf_path = temp_pdf.name
        
        # Download PDF from the direct URL
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(response.content)

        # Load PDF using LangChain
        loader = PyPDFLoader(temp_pdf_path)
        documents_data = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        docs = text_splitter.split_documents(documents_data)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Store vectorstore for this document
        vectorstores[document_id] = vectorstore

        return True, len(docs)

    except Exception as e:
        raise Exception(f"Error processing PDF with LangChain: {str(e)}")
    finally:
        # Clean up the temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

# In the get_ai_response function in app.py
def get_ai_response(query, document_id):
    """Get AI response using LangChain QA chain"""
    try:
        if document_id not in vectorstores:
            return "Document not processed yet. Please try again."

        # Create QA chain
        vectorstore = vectorstores[document_id]
        retriever = vectorstore.as_retriever()
        # Replace ChatOpenAI with ChatGroq
        llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        # Get response
        result = qa_chain.invoke({"query": query})
        return result["result"]

    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'data': {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        }
    })

    
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'data': {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'openai_configured': bool(os.getenv('OPENAI_API_KEY'))
        }
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload PDF file endpoint"""
    try:
        # Check if OpenAI API key is configured
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({
                'success': False,
                'error': 'OPENAI_API_KEY is not set'
            }), 500

        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Only PDF files are allowed'
            }), 400

        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"{document_id}_{filename}")
        file.save(temp_path)

        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE:
            os.remove(temp_path)
            return jsonify({
                'success': False,
                'error': 'File size too large. Maximum size is 16MB'
            }), 400

        # Upload to tmpfiles.org
        try:
            public_url = upload_to_tmpfiles(temp_path)
        except Exception as e:
            os.remove(temp_path)
            return jsonify({
                'success': False,
                'error': f'Error uploading file: {str(e)}'
            }), 500

        # Process PDF with LangChain
        success, chunk_count = process_pdf_with_langchain(public_url, document_id)

        # Store document information
        documents[document_id] = {
            'id': document_id,
            'filename': filename,
            'public_url': public_url,
            'upload_time': datetime.now().isoformat(),
            'size': file_size,
            'chunk_count': chunk_count
        }
        
        # Initialize chat history
        chat_histories[document_id] = []

        # Clean up temporary file
        os.remove(temp_path)

        return jsonify({
            'success': True,
            'data': {
                'document_id': document_id,
                'filename': filename,
                'upload_time': documents[document_id]['upload_time'],
                'chunk_count': chunk_count
            }
        })

    except Exception as e:
        # Ensure temporary file is cleaned up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            'success': False,
            'error': f'Error processing file: {str(e)}'
        }), 500

@app.route('/chat/<document_id>', methods=['POST'])
def chat_with_document(document_id):
    """Chat with document endpoint"""
    try:
        if document_id not in documents:
            return jsonify({
                'success': False,
                'error': 'Document not found'
            }), 404

        if document_id not in vectorstores:
            return jsonify({
                'success': False,
                'error': 'Document not processed yet'
            }), 400

        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400

        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            }), 400

        # Get AI response using LangChain
        ai_response = get_ai_response(user_message, document_id)
        
        # Store chat history
        chat_entry = {
            'id': str(uuid.uuid4()),
            'user_message': user_message,
            'ai_response': ai_response,
            'timestamp': datetime.now().isoformat()
        }
        
        if document_id not in chat_histories:
            chat_histories[document_id] = []
        
        chat_histories[document_id].append(chat_entry)

        return jsonify({
            'success': True,
            'data': {
                'response': ai_response,
                'timestamp': chat_entry['timestamp']
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing chat: {str(e)}'
        }), 500

@app.route('/chat/<document_id>/history', methods=['GET'])
def get_chat_history(document_id):
    """Get chat history for a document"""
    try:
        if document_id not in documents:
            return jsonify({
                'success': False,
                'error': 'Document not found'
            }), 404

        history = chat_histories.get(document_id, [])
        
        return jsonify({
            'success': True,
            'data': {
                'document_id': document_id,
                'chat_history': history
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error retrieving chat history: {str(e)}'
        }), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    try:
        document_list = []
        for doc_id, doc_info in documents.items():
            document_list.append({
                'id': doc_id,
                'filename': doc_info['filename'],
                'upload_time': doc_info['upload_time'],
                'size': doc_info['size'],
                'chunk_count': doc_info.get('chunk_count', 0)
            })
        
        return jsonify({
            'success': True,
            'data': {
                'documents': document_list,
                'total': len(document_list)
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error retrieving documents: {str(e)}'
        }), 500

@app.route('/documents/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    """Delete a document and its chat history"""
    try:
        if document_id not in documents:
            return jsonify({
                'success': False,
                'error': 'Document not found'
            }), 404

        # Remove from storage
        del documents[document_id]
        
        # Remove chat history
        if document_id in chat_histories:
            del chat_histories[document_id]
            
        # Remove vectorstore
        if document_id in vectorstores:
            del vectorstores[document_id]

        return jsonify({
            'success': True,
            'data': {
                'message': 'Document deleted successfully'
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error deleting document: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("WARNING: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key'")
    
    app.run(debug=True, host='0.0.0.0', port=5000)