"""
AIML Document Analysis Application
A Streamlit-based application for document analysis and Q&A using LLM models with vector embeddings.
"""

import streamlit as st
import asyncio
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import tempfile
import hashlib

# File processing
try:
    from PyPDF2 import PdfReader
    import docx
except ImportError:
    st.error("Required libraries not installed. Please install: PyPDF2, python-docx")
    st.stop()

# Vector embeddings and database
try:
    from sentence_transformers import SentenceTransformer
    from chromadb.config import Settings
    import chromadb
except ImportError:
    st.error("Required libraries not installed. Please install: sentence-transformers, chromadb")
    st.stop()

# LLM and Langchain
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import requests
except ImportError:
    st.error("Required libraries not installed. Please install: langchain, requests")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"  # Free, lightweight embeddings model
DEFAULT_LLM_API = "https://api.groq.com/openai/v1/chat/completions"  # Groq - Free API
DEFAULT_LLM_MODEL = "mixtral-8x7b-32768"  # Free model from Groq
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONTEXT_LENGTH = 5000
LOGS_FILE = "query_logs.jsonl"
DB_PATH = "./chroma_db"

# ==================== SESSION STATE INITIALIZATION ====================
def initialize_session_state():
    """Initialize all session state variables."""
    if 'embeddings_model' not in st.session_state:
        st.session_state.embeddings_model = None
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    if 'file_hash' not in st.session_state:
        st.session_state.file_hash = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

initialize_session_state()

# ==================== FILE PROCESSING ====================
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file."""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF: {str(e)}")
        raise Exception(f"Failed to extract PDF: {str(e)}")

def extract_text_from_docx(docx_file) -> str:
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX: {str(e)}")
        raise Exception(f"Failed to extract DOCX: {str(e)}")

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file based on file type."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            return extract_text_from_pdf(uploaded_file)
        elif file_extension in ['docx', 'doc']:
            return extract_text_from_docx(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        logger.error(f"Error extracting file: {str(e)}")
        raise

def get_file_hash(content: str) -> str:
    """Generate hash of file content."""
    return hashlib.md5(content.encode()).hexdigest()

# ==================== TEXT CHUNKING ====================
def chunk_text(text: str) -> List[str]:
    """Split text into chunks."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise

# ==================== EMBEDDINGS AND VECTOR DB ====================
def load_embeddings_model():
    """Load the embeddings model."""
    try:
        if st.session_state.embeddings_model is None:
            with st.spinner("Loading embeddings model..."):
                st.session_state.embeddings_model = SentenceTransformer(EMBEDDINGS_MODEL)
        return st.session_state.embeddings_model
    except Exception as e:
        logger.error(f"Error loading embeddings model: {str(e)}")
        raise

def initialize_vector_db():
    """Initialize Chroma vector database."""
    try:
        if st.session_state.vector_db is None:
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=DB_PATH,
                anonymized_telemetry=False
            )
            st.session_state.vector_db = chromadb.Client(settings)
        return st.session_state.vector_db
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}")
        raise

def process_and_store_document(text: str, file_name: str):
    """Process document and store embeddings in vector database."""
    try:
        # Load models
        embeddings_model = load_embeddings_model()
        vector_db = initialize_vector_db()
        
        # Create or get collection
        collection_name = "documents"
        try:
            st.session_state.collection = vector_db.get_collection(collection_name)
            st.session_state.collection.delete(where={"source": file_name})
        except:
            st.session_state.collection = vector_db.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Chunk text
        chunks = chunk_text(text)
        
        # Generate embeddings and store
        with st.spinner("Generating embeddings..."):
            for i, chunk in enumerate(chunks):
                try:
                    embedding = embeddings_model.encode(chunk).tolist()
                    st.session_state.collection.add(
                        ids=[f"{file_name}_chunk_{i}"],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "source": file_name,
                            "chunk_index": i,
                            "timestamp": datetime.now().isoformat()
                        }]
                    )
                except Exception as e:
                    logger.warning(f"Error storing chunk {i}: {str(e)}")
                    continue
        
        logger.info(f"Successfully processed {file_name} with {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

# ==================== RETRIEVAL ====================
def retrieve_context(query: str, num_results: int = 3) -> str:
    """Retrieve relevant context from vector database."""
    try:
        if st.session_state.collection is None:
            return ""
        
        embeddings_model = load_embeddings_model()
        query_embedding = embeddings_model.encode(query).tolist()
        
        results = st.session_state.collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results
        )
        
        context = ""
        if results and results['documents']:
            for doc_list in results['documents']:
                for doc in doc_list:
                    context += doc + "\n\n"
        
        return context[:MAX_CONTEXT_LENGTH]
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return ""

# ==================== LLM INTERACTION ====================
async def query_llm_async(
    query: str,
    context: str,
    api_endpoint: str,
    api_key: str,
    model_name: str
) -> Tuple[str, bool]:
    """Query LLM with context asynchronously."""
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            query_llm_sync,
            query,
            context,
            api_endpoint,
            api_key,
            model_name
        )
        return response, True
    except Exception as e:
        logger.error(f"Error in async LLM query: {str(e)}")
        return str(e), False

def query_llm_sync(
    query: str,
    context: str,
    api_endpoint: str,
    api_key: str,
    model_name: str
) -> str:
    """Query LLM synchronously."""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are a helpful assistant. Answer the user's question based on the provided context. 
If the context doesn't contain relevant information, say so. Be concise and accurate."""
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(api_endpoint, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise ValueError("Unexpected response format from LLM")
            
    except requests.exceptions.Timeout:
        raise Exception("LLM request timed out. Please check your API endpoint.")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to LLM API. Please check your API endpoint.")
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise Exception("Invalid API key. Please check your credentials.")
        elif response.status_code == 429:
            raise Exception("Rate limit exceeded. Please try again later.")
        else:
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error querying LLM: {str(e)}")
        raise

# ==================== LOGGING ====================
def log_query(query: str, response: str, file_name: str, model_used: str, api_endpoint: str):
    """Log user query and response."""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "file_name": file_name,
            "query": query,
            "response": response,
            "model": model_used,
            "api_endpoint": api_endpoint
        }
        with open(LOGS_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.info(f"Logged query: {query[:50]}...")
    except Exception as e:
        logger.error(f"Error logging query: {str(e)}")

# ==================== STREAMLIT UI ====================
def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Document AI Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 30px;
        }
        .section-header {
            color: #ff7f0e;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .info-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .error-box {
            background-color: #ffcccc;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            color: #cc0000;
        }
        .success-box {
            background-color: #ccffcc;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            color: #009900;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown("<h1 class='main-header'>📄 Document AI Assistant</h1>", unsafe_allow_html=True)
    
    # Create two columns for top section
    col1, col2, col3 = st.columns(3, gap="medium")
    
    # ==================== FILE UPLOAD SECTION ====================
    with col1:
        st.markdown("<p class='section-header'>📁 Upload Document</p>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Upload a PDF or DOCX file for analysis</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'doc'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            st.success(f"✓ File selected: {uploaded_file.name}")
    
    # ==================== API CONFIGURATION SECTION ====================
    with col2:
        st.markdown("<p class='section-header'>🔑 API Configuration</p>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Enter your LLM API endpoint (optional - uses free default if blank)</div>", unsafe_allow_html=True)
        api_endpoint = st.text_input(
            "API Endpoint",
            value="",
            placeholder="https://api.openai.com/v1/chat/completions",
            label_visibility="collapsed"
        ).strip()
        
        if api_endpoint:
            st.info(f"Using custom API: {api_endpoint}")
        else:
            st.info(f"Using default free API: {DEFAULT_LLM_API}")
    
    # ==================== API KEY SECTION ====================
    with col3:
        st.markdown("<p class='section-header'>🔐 API Key</p>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Enter your API key (optional when using default)</div>", unsafe_allow_html=True)
        api_key = st.text_input(
            "API Key",
            value="",
            type="password",
            placeholder="Enter API key here",
            label_visibility="collapsed"
        ).strip()
        
        if api_key:
            st.success("✓ API Key entered (hidden)")
        elif not api_endpoint:
            st.info("✓ Using default free model (no key needed)")
    
    # ==================== DIVIDER ====================
    st.markdown("---")
    
    # ==================== DOCUMENT PROCESSING ====================
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read()
            file_hash = get_file_hash(file_content.decode() if isinstance(file_content, bytes) else file_content)
            
            # Check if file needs reprocessing
            if st.session_state.file_hash != file_hash:
                with st.spinner("Processing document..."):
                    # Extract text
                    uploaded_file.seek(0)
                    text = extract_text_from_file(uploaded_file)
                    
                    if not text.strip():
                        st.error("❌ No text found in the document. Please check the file.")
                    else:
                        # Process and store
                        process_and_store_document(text, uploaded_file.name)
                        st.session_state.file_processed = True
                        st.session_state.file_hash = file_hash
                        st.success("✓ Document processed and indexed successfully!")
                        logger.info(f"Document processed: {uploaded_file.name}")
            else:
                if st.session_state.file_processed:
                    st.success("✓ Document already processed")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger.error(f"File processing error: {str(e)}")
            st.session_state.file_processed = False
    else:
        st.warning("⚠️ Please upload a file to proceed")
    
    # ==================== QUERY SECTION ====================
    st.markdown("---")
    st.markdown("<p class='section-header'>💬 Ask a Question</p>", unsafe_allow_html=True)
    
    if st.session_state.file_processed:
        # Prepare API credentials
        use_custom_api = bool(api_endpoint and api_key)
        
        if use_custom_api:
            llm_api = api_endpoint
            llm_key = api_key
            llm_model = st.text_input(
                "Model Name (for custom API)",
                value="gpt-3.5-turbo",
                help="Enter the model name to use from your API"
            ).strip()
        else:
            if not api_endpoint and not api_key:
                llm_api = DEFAULT_LLM_API
                llm_key = st.secrets.get("GROQ_API_KEY", "")
                llm_model = DEFAULT_LLM_MODEL
                
                if not llm_key:
                    st.warning("⚠️ No Groq API key found. Set GROQ_API_KEY in secrets.toml or provide custom API credentials.")
                    st.info("Get a free API key at: https://console.groq.com")
            else:
                st.error("❌ Please provide both API endpoint and key, or leave both blank to use the default.")
                return
        
        # Query input
        query = st.text_input(
            "Enter your question",
            placeholder="What is the document about?",
            label_visibility="collapsed"
        ).strip()
        
        if query:
            try:
                # Retrieve context
                context = retrieve_context(query)
                
                if not context:
                    st.warning("⚠️ No relevant information found in the document for this query.")
                    st.info("Try rephrasing your question or upload a different document.")
                else:
                    # Query LLM
                    with st.spinner("🔄 Fetching response..."):
                        response, success = asyncio.run(
                            query_llm_async(
                                query,
                                context,
                                llm_api,
                                llm_key,
                                llm_model
                            )
                        )
                    
                    if success:
                        st.markdown("<div class='success-box'>✓ Response Generated</div>", unsafe_allow_html=True)
                        st.markdown("### Response:")
                        st.write(response)
                        
                        # Log the interaction
                        log_query(
                            query,
                            response,
                            uploaded_file.name,
                            llm_model,
                            llm_api
                        )
                        
                    else:
                        st.error(f"❌ Error: {response}")
                        logger.error(f"LLM query failed: {response}")
                        
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ An error occurred: {error_msg}")
                logger.error(f"Query processing error: {error_msg}")
    else:
        st.info("📤 Please upload and process a document first to ask questions.")
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Document AI Assistant** - An intelligent document analysis tool powered by LLMs.
        
        **Features:**
        - 📄 Support for PDF and DOCX files
        - 🔍 Vector-based semantic search
        - 🤖 Multiple LLM model support
        - 📊 Query logging and history
        - ⚡ Async processing for multi-user support
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        if st.session_state.file_processed and uploaded_file:
            try:
                text = uploaded_file.getvalue().decode() if isinstance(uploaded_file.getvalue(), bytes) else uploaded_file.getvalue()
                token_estimate = len(text.split())
                st.metric("Estimated Tokens", token_estimate)
                st.metric("File Name", uploaded_file.name)
            except:
                pass
        
        # Log viewer
        st.markdown("---")
        st.markdown("### 📋 Query Logs")
        if st.button("View Logs"):
            try:
                if os.path.exists(LOGS_FILE):
                    with open(LOGS_FILE, 'r') as f:
                        logs = f.readlines()
                    if logs:
                        st.json([json.loads(log) for log in logs[-5:]])  # Show last 5
                    else:
                        st.info("No queries logged yet.")
                else:
                    st.info("No logs file found yet.")
            except Exception as e:
                st.error(f"Error reading logs: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 🔧 Configuration")
        st.markdown(f"""
        **Embeddings Model:** {EMBEDDINGS_MODEL}
        **Chunk Size:** {CHUNK_SIZE}
        **Default LLM Model:** {DEFAULT_LLM_MODEL}
        """)

if __name__ == "__main__":
    main()
