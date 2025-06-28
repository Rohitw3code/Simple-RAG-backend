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
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
import shutil
import openai

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
AVAILABLE_GROQ_MODELS = ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']  # Supported Groq models

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for demo purposes
documents = {}
chat_histories = {}
vectorstores = {}
document_models = {}  # Store selected model per document

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
            url = response_json['data']['url']
            return url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        else:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")
    except Exception as e:
        raise Exception(f"Error uploading to tmpfiles: {str(e)}")

def process_pdf_with_langchain(pdf_url, document_id, openai_api_key=None):
    """Process PDF using LangChain and create vector store"""
    temp_pdf_path = None
    try:
        # Set OpenAI API key for this operation
        original_api_key = os.environ.get('OPENAI_API_KEY')
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key

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
        try:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
        except openai.AuthenticationError:
            raise Exception("Invalid OpenAI API key provided")
        
        # Store vectorstore for this document
        vectorstores[document_id] = vectorstore

        # Restore original API key
        if openai_api_key and original_api_key:
            os.environ['OPENAI_API_KEY'] = original_api_key
        elif openai_api_key and not original_api_key:
            del os.environ['OPENAI_API_KEY']

        return True, len(docs)

    except Exception as e:
        raise Exception(f"Error processing PDF with LangChain: {str(e)}")
    finally:
        # Clean up the temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

def combine_vectorstores(document_ids):
    """Combine vector stores for multiple documents"""
    try:
        if not document_ids:
            raise Exception("No document IDs provided")

        valid_vectorstores = []
        for doc_id in document_ids:
            if doc_id in vectorstores:
                valid_vectorstores.append(vectorstores[doc_id])
            else:
                raise Exception(f"Vector store for document {doc_id} not found")

        if not valid_vectorstores:
            raise Exception("No valid vector stores found")

        # Merge vector stores using FAISS merge_from
        combined_vectorstore = valid_vectorstores[0]
        for vs in valid_vectorstores[1:]:
            combined_vectorstore.merge_from(vs)

        return combined_vectorstore

    except Exception as e:
        raise Exception(f"Error combining vector stores: {str(e)}")

def get_ai_response(query, document_ids, groq_api_key=None):
    """Get AI response using LangChain QA chain for multiple documents"""
    try:
        # Combine vector stores for the provided document IDs
        combined_vectorstore = combine_vectorstores(document_ids)
        
        # Set Groq API key for this operation
        original_groq_key = os.environ.get('GROQ_API_KEY')
        if groq_api_key:
            os.environ['GROQ_API_KEY'] = groq_api_key

        # Create QA chain
        retriever = combined_vectorstore.as_retriever()
        model_name = document_models.get(document_ids[0], 'llama3-8b-8192')  # Use primary document's model
        try:
            llm = ChatGroq(temperature=0, model_name=model_name)
        except Exception as e:
            raise Exception("Invalid Groq API key or model configuration")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        # Get response
        result = qa_chain.invoke({"query": query})

        # Restore original Groq API key
        if groq_api_key and original_groq_key:
            os.environ['GROQ_API_KEY'] = original_groq_key
        elif groq_api_key and not original_groq_key:
            del os.environ['GROQ_API_KEY']

        return result["result"]

    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/', methods=['GET'])
def star():
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
            'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
            'groq_configured': bool(os.getenv('GROQ_API_KEY'))
        }
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload multiple PDF files endpoint"""
    try:
        # Check for API keys in request
        data = request.form.to_dict()
        openai_api_key = data.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
        if not openai_api_key:
            return jsonify({
                'success': False,
                'error': 'OPENAI_API_KEY is required (either in request or environment)'
            }), 500

        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        uploaded_files = request.files.getlist('file')
        if not uploaded_files:
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400

        results = []
        for file in uploaded_files:
            if file.filename == '':
                continue

            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Only PDF files are allowed'
                })
                continue

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
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': 'File size too large. Maximum size is 16MB'
                })
                continue

            # Upload to tmpfiles.org
            try:
                public_url = upload_to_tmpfiles(temp_path)
            except Exception as e:
                os.remove(temp_path)
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': f'Error uploading file: {str(e)}'
                })
                continue

            # Process PDF with LangChain
            try:
                success, chunk_count = process_pdf_with_langchain(public_url, document_id, openai_api_key)
            except Exception as e:
                os.remove(temp_path)
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': f'Error processing PDF: {str(e)}'
                })
                continue

            # Store document information
            documents[document_id] = {
                'id': document_id,
                'filename': filename,
                'public_url': public_url,
                'upload_time': datetime.now().isoformat(),
                'size': file_size,
                'chunk_count': chunk_count
            }
            
            # Initialize chat history and default model
            chat_histories[document_id] = []
            document_models[document_id] = 'llama3-8b-8192'  # Default model

            # Clean up temporary file
            os.remove(temp_path)

            results.append({
                'filename': filename,
                'success': True,
                'data': {
                    'document_id': document_id,
                    'filename': filename,
                    'upload_time': documents[document_id]['upload_time'],
                    'chunk_count': chunk_count
                }
            })

        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'total_uploaded': len([r for r in results if r['success']])
            }
        })

    except Exception as e:
        # Ensure temporary file is cleaned up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({
            'success': False,
            'error': f'Error processing files: {str(e)}'
        }), 500

@app.route('/chat/<document_id>', methods=['POST'])
def chat_with_document(document_id):
    """Chat with one or more documents endpoint"""
    try:
        if document_id not in documents:
            return jsonify({
                'success': False,
                'error': 'Primary document not found'
            }), 404

        if document_id not in vectorstores:
            return jsonify({
                'success': False,
                'error': 'Primary document not processed yet'
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

        # Get Groq API key from request or environment
        groq_api_key = data.get('groq_api_key', os.getenv('GROQ_API_KEY'))
        if not groq_api_key:
            return jsonify({
                'success': False,
                'error': 'GROQ_API_KEY is required (either in request or environment)'
            }), 500

        # Get additional document IDs from query parameter or JSON body
        doc_ids = data.get('doc_ids', [])
        if isinstance(doc_ids, str):
            doc_ids = [doc_id.strip() for doc_id in doc_ids.split(',')]
        document_ids = [document_id] + [doc_id for doc_id in doc_ids if doc_id != document_id]
        
        # Validate additional document IDs
        for doc_id in document_ids:
            if doc_id not in documents:
                return jsonify({
                    'success': False,
                    'error': f'Document {doc_id} not found'
                }), 404
            if doc_id not in vectorstores:
                return jsonify({
                    'success': False,
                    'error': f'Document {doc_id} not processed yet'
                }), 400

        # Get AI response using LangChain
        ai_response = get_ai_response(user_message, document_ids, groq_api_key)
        
        # Store chat history under primary document_id
        chat_entry = {
            'id': str(uuid.uuid4()),
            'user_message': user_message,
            'ai_response': ai_response,
            'document_ids': document_ids,
            'timestamp': datetime.now().isoformat()
        }
        
        if document_id not in chat_histories:
            chat_histories[document_id] = []
        
        chat_histories[document_id].append(chat_entry)

        return jsonify({
            'success': True,
            'data': {
                'response': ai_response,
                'document_ids': document_ids,
                'timestamp': chat_entry['timestamp']
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing chat: {str(e)}'
        }), 500

@app.route('/set-model/<document_id>', methods=['POST'])
def set_model(document_id):
    """Set Groq model for a specific document"""
    try:
        if document_id not in documents:
            return jsonify({
                'success': False,
                'error': 'Document not found'
            }), 404

        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({
                'success': False,
                'error': 'Model name is required'
            }), 400

        model_name = data['model_name'].strip()
        if not model_name:
            return jsonify({
                'success': False,
                'error': 'Model name cannot be empty'
            }), 400

        if model_name not in AVAILABLE_GROQ_MODELS:
            return jsonify({
                'success': False,
                'error': f'Invalid model name. Available models: {", ".join(AVAILABLE_GROQ_MODELS)}'
            }), 400

        # Store the selected model
        document_models[document_id] = model_name

        return jsonify({
            'success': True,
            'data': {
                'document_id': document_id,
                'model_name': model_name,
                'message': f'Model set to {model_name} for document {document_id}'
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error setting model: {str(e)}'
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
                'chunk_count': doc_info.get('chunk_count', 0),
                'model': document_models.get(doc_id, 'llama3-8b-8192')
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

        # Remove model selection
        if document_id in document_models:
            del document_models[document_id]

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
    # Check for API keys
    if not os.getenv('OPENAI_API_KEY') or not os.getenv('GROQ_API_KEY'):
        print("WARNING: API keys not set in environment variables!")
        print("You can set them or provide them in API requests.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)