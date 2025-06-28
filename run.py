#!/usr/bin/env python3
"""
Simple script to run the Flask API server
"""
import os
import sys
from app import app

if __name__ == '__main__':
    # Check for required API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')

    if not openai_key or not groq_key:
        print("❌ ERROR: Missing required environment variables!")
        print("\n📝 This application requires two API keys to function:")
        
        if not openai_key:
            print("\n1. OpenAI API Key (for embeddings)")
            print("   - Get your key from: https://platform.openai.com/api-keys")
            print("   - Set the environment variable:")
            print("     export OPENAI_API_KEY='your-api-key-here'")

        if not groq_key:
            print("\n2. Groq API Key (for the chat model)")
            print("   - Get your key from: https://console.groq.com/keys")
            print("   - Set the environment variable:")
            print("     export GROQ_API_KEY='your-api-key-here'")
        
        print("\n⚠️ The API will not work without both keys!")
        sys.exit(1)
    
    print("🚀 Starting Flask API server...")
    print("📡 API will be available at: http://localhost:5000")
    print("🔑 OpenAI API key for embeddings configured ✅")
    print("🔑 Groq API key for chat configured ✅")
    print("\n📋 Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /upload - Upload PDF")
    print("  POST /chat/<doc_id> - Chat with document")
    print("  GET  /chat/<doc_id>/history - Get chat history")
    print("  GET  /documents - List documents")
    print("  DELETE /documents/<doc_id> - Delete document")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)