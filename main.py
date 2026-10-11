import streamlit as st
import os
import tempfile
import json
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import sqlite3
from datetime import datetime
import subprocess
import sys
import hashlib
import re

# Install faiss-cpu for Windows compatibility
try:
    import faiss
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    import faiss

# Data models for better organization
@dataclass
class Recipe:
    id: str
    name: str
    description: str
    prep_time: int
    cook_time: int
    servings: int
    ingredients: List[Dict[str, str]]
    instructions: List[str]
    nutritional_info: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'prep_time': self.prep_time,
            'cook_time': self.cook_time,
            'servings': self.servings,
            'ingredients': self.ingredients,
            'instructions': self.instructions,
            'nutritional_info': self.nutritional_info,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recipe':
        return cls(**data)

@dataclass
class Session:
    id: str
    name: str
    created_at: str
    updated_at: str
    recipes: List[Recipe]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'recipes': [recipe.to_dict() for recipe in self.recipes]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        recipes = [Recipe.from_dict(recipe) for recipe in data.get('recipes', [])]
        return cls(
            id=data['id'],
            name=data['name'],
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            recipes=recipes
        )

# Configuration constants
class Config:
    EMBEDDING_MODEL_LOCAL = "BAAI/bge-small-en-v1.5"
    DEFAULT_OLLAMA_MODELS = ["llama3:8b", "mistral", "phi3", "gemma:2b", "nomic-embed-text"]
    TEMPERATURE = 0.7
    MAX_TOKENS = 4096
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    SUPPORTED_FILE_TYPES = ["pdf", "txt", "md"]
    DB_FILE = "culinary_sessions.db"

# Initialize session state
def init_session_state():
    defaults = {
        'vector_store': None,
        'current_session': None,
        'chat_history': [],
        'ollama_models': [],
        'selected_model': "",
        'embedding_model': Config.EMBEDDING_MODEL_LOCAL,
        'ollama_url': "http://localhost:11434",
        'ollama_checked': False,
        'active_tab': 'workspace',
        'processing_state': {
            'status': 'idle',
            'message': '',
            'progress': 0,
            'start_time': None
        },
        'document_cache': {},
        'sessions': [],
        'search_query': '',
        'search_results': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Database functions for session management
def init_db():
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        data TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

def save_session(session: Session) -> bool:
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    
    session.updated_at = datetime.now().isoformat()
    session_data = json.dumps(session.to_dict())
    
    cursor.execute('''
    INSERT OR REPLACE INTO sessions (id, name, created_at, updated_at, data)
    VALUES (?, ?, ?, ?, ?)
    ''', (session.id, session.name, session.created_at, session.updated_at, session_data))
    
    conn.commit()
    conn.close()
    
    return True

def load_session(session_id: str) -> Optional[Session]:
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('SELECT data FROM sessions WHERE id = ?', (session_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        try:
            session_data = json.loads(result[0])
            return Session.from_dict(session_data)
        except Exception as e:
            st.error(f"Error loading session: {str(e)}")
            return None
    
    return None

def delete_session(session_id: str) -> bool:
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    affected_rows = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    return affected_rows > 0

def search_sessions(query: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, name, created_at, updated_at FROM sessions
    WHERE name LIKE ?
    ORDER BY updated_at DESC
    ''', (f'%{query}%',))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'id': row[0],
            'name': row[1],
            'created_at': row[2],
            'updated_at': row[3]
        })
    
    conn.close()
    return results

def get_all_sessions() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(Config.DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, name, created_at, updated_at FROM sessions
    ORDER BY updated_at DESC
    ''')
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'id': row[0],
            'name': row[1],
            'created_at': row[2],
            'updated_at': row[3]
        })
    
    conn.close()
    return results

# Initialize database
init_db()

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Utility functions
def generate_id() -> str:
    return hashlib.md5(f"{time.time()}{os.urandom(8)}".encode()).hexdigest()[:8]

def update_processing_state(status: str, message: str = "", progress: int = 0):
    st.session_state.processing_state = {
        'status': status,
        'message': message,
        'progress': progress,
        'start_time': datetime.now() if status == 'processing' else st.session_state.processing_state.get('start_time')
    }
    if status == 'complete':
        st.session_state.processing_state = {'status': 'idle', 'message': '', 'progress': 0, 'start_time': None}
    elif status == 'error':
        st.session_state.processing_state['start_time'] = None

def check_ollama_connection(url: str) -> bool:
    try:
        response = requests.get(f"{url}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def fetch_ollama_models(url: str) -> List[str]:
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except Exception:
        pass
    return []

def initialize_local_embeddings(model_name: str = Config.EMBEDDING_MODEL_LOCAL):
    if model_name in st.session_state.document_cache:
        return st.session_state.document_cache[model_name]
    
    try:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        st.session_state.document_cache[model_name] = embeddings
        return embeddings
    except Exception:
        if Config.EMBEDDING_MODEL_LOCAL in st.session_state.document_cache:
            return st.session_state.document_cache[Config.EMBEDDING_MODEL_LOCAL]
        else:
            embeddings = HuggingFaceBgeEmbeddings(model_name=Config.EMBEDDING_MODEL_LOCAL)
            st.session_state.document_cache[Config.EMBEDDING_MODEL_LOCAL] = embeddings
            return embeddings

def initialize_rag_system(documents, embedding_model: str):
    update_processing_state('processing', 'Processing documents...', 10)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len
    )
    
    update_processing_state('processing', 'Splitting documents...', 30)
    chunks = text_splitter.split_documents(documents)
    
    update_processing_state('processing', 'Creating embeddings...', 60)
    embeddings = initialize_local_embeddings(embedding_model)
    
    try:
        test_embedding = embeddings.embed_query("test")
        dimension = len(test_embedding)
        index = faiss.IndexFlatL2(dimension)
        
        update_processing_state('processing', 'Building vector store...', 80)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )
        
        vector_store.add_documents(chunks)
        update_processing_state('complete', f'✅ Processed {len(chunks)} chunks from {len(documents)} files', 100)
        return vector_store
    except Exception as e:
        update_processing_state('error', f'Failed to create vector store: {str(e)}')
        return None

def search_knowledge_base(query: str, k: int = 3) -> List[str]:
    if not st.session_state.vector_store:
        return []
    
    try:
        docs = st.session_state.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    except Exception:
        return []

def generate_recipe_with_ollama(prompt: str, model_name: str, ollama_url: str) -> Optional[str]:
    try:
        # Get context from knowledge base if available
        kb_context = ""
        if st.session_state.vector_store:
            kb_results = search_knowledge_base(prompt, k=2)
            if kb_results:
                kb_context = "\n\nKnowledge Base Context:\n" + "\n".join(kb_results[:2])
        
        # Enhanced prompt for recipe generation
        enhanced_prompt = f"""
        You are a professional chef. Create a detailed recipe based on the following request.
        Include ingredients with precise measurements, step-by-step instructions, prep time, cook time, and servings.
        Also provide basic nutritional information (calories, protein, carbs, fat).
        
        {kb_context}
        
        User Request: {prompt}
        
        Recipe:
        """
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": {
                    "temperature": Config.TEMPERATURE,
                    "num_predict": Config.MAX_TOKENS
                }
            }
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        return None
    except Exception as e:
        st.error(f"Error generating recipe: {str(e)}")
        return None

def parse_recipe_from_text(text: str) -> Optional[Recipe]:
    """Parse recipe information from text and return a Recipe object"""
    try:
        # Initialize recipe with basic info
        recipe_id = generate_id()
        now = datetime.now().isoformat()
        
        # Extract title
        title_match = re.search(r'(?:title|name)[:\s]+([^\n]+)', text, re.IGNORECASE)
        name = title_match.group(1).strip() if title_match else "Untitled Recipe"
        
        # Extract description
        desc_match = re.search(r'(?:description|about)[:\s]+([^\n]+(?:\n[^#\n]+)*)', text, re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else ""
        
        # Extract prep time
        prep_match = re.search(r'(?:prep|preparation)(?:\s+time)?[:\s]+(\d+)(?:\s+minutes?)?', text, re.IGNORECASE)
        prep_time = int(prep_match.group(1)) if prep_match else 0
        
        # Extract cook time
        cook_match = re.search(r'(?:cook|cooking)(?:\s+time)?[:\s]+(\d+)(?:\s+minutes?)?', text, re.IGNORECASE)
        cook_time = int(cook_match.group(1)) if cook_match else 0
        
        # Extract servings
        servings_match = re.search(r'(?:servings|serves|yields?)[:\s]+(\d+)', text, re.IGNORECASE)
        servings = int(servings_match.group(1)) if servings_match else 1
        
        # Extract ingredients
        ingredients = []
        ingredients_section = re.search(r'(?:ingredients)[:\s\n]+([^\n#]+(?:\n[^#\n]+)*)', text, re.IGNORECASE)
        if ingredients_section:
            ingredient_lines = ingredients_section.group(1).strip().split('\n')
            for line in ingredient_lines:
                if line.strip():
                    # Try to parse amount and name
                    amount_match = re.match(r'^([\d\/\s\.]+(?:\s+(?:cup|cups|tbsp|tsp|oz|lb|g|kg|ml|l)?)?)\s+(.+)$', line.strip())
                    if amount_match:
                        amount = amount_match.group(1).strip()
                        name = amount_match.group(2).strip()
                    else:
                        amount = ""
                        name = line.strip()
                    
                    ingredients.append({"name": name, "amount": amount})
        
        # Extract instructions
        instructions = []
        instructions_section = re.search(r'(?:instructions|directions|steps|method)[:\s\n]+([^\n#]+(?:\n[^#\n]+)*)', text, re.IGNORECASE)
        if instructions_section:
            instruction_lines = instructions_section.group(1).strip().split('\n')
            for line in instruction_lines:
                if line.strip():
                    # Try to extract step number
                    step_num_match = re.match(r'^(\d+)[\.\)]\s*(.+)$', line.strip())
                    if step_num_match:
                        instruction = step_num_match.group(2).strip()
                    else:
                        instruction = line.strip()
                    
                    instructions.append(instruction)
        
        # Extract nutritional information
        nutritional_info = {}
        nutrition_section = re.search(r'(?:nutrition|nutritional\s+information)[:\s\n]+([^\n#]+(?:\n[^#\n]+)*)', text, re.IGNORECASE)
        if nutrition_section:
            nutrition_text = nutrition_section.group(1).strip()
            
            # Extract calories
            calories_match = re.search(r'(?:calories|energy)[:\s]+(\d+)', nutrition_text, re.IGNORECASE)
            if calories_match:
                nutritional_info["calories"] = int(calories_match.group(1))
            
            # Extract protein
            protein_match = re.search(r'(?:protein)[:\s]+(\d+(?:\.\d+)?)\s*g', nutrition_text, re.IGNORECASE)
            if protein_match:
                nutritional_info["protein"] = float(protein_match.group(1))
            
            # Extract carbs
            carbs_match = re.search(r'(?:carbs|carbohydrates)[:\s]+(\d+(?:\.\d+)?)\s*g', nutrition_text, re.IGNORECASE)
            if carbs_match:
                nutritional_info["carbs"] = float(carbs_match.group(1))
            
            # Extract fat
            fat_match = re.search(r'(?:fat)[:\s]+(\d+(?:\.\d+)?)\s*g', nutrition_text, re.IGNORECASE)
            if fat_match:
                nutritional_info["fat"] = float(fat_match.group(1))
        
        # Create recipe object
        recipe = Recipe(
            id=recipe_id,
            name=name,
            description=description,
            prep_time=prep_time,
            cook_time=cook_time,
            servings=servings,
            ingredients=ingredients,
            instructions=instructions,
            nutritional_info=nutritional_info,
            created_at=now
        )
        
        return recipe
    except Exception as e:
        st.error(f"Error parsing recipe: {str(e)}")
        return None

# UI Components
def render_processing_indicator():
    """Render a processing indicator with progress"""
    if st.session_state.processing_state['status'] == 'processing':
        progress = st.session_state.processing_state['progress'] / 100
        message = st.session_state.processing_state['message']
        
        st.markdown(f"""
        <div style="padding: 1rem; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #4CAF50;">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <div class="processing-spinner" style="margin-right: 0.5rem;">⚡</div>
                <strong>Processing...</strong>
            </div>
            <div style="margin-bottom: 0.5rem;">{message}</div>
            <div style="background-color: #e0e0e0; border-radius: 4px; height: 8px;">
                <div style="background-color: #4CAF50; height: 100%; width: {progress * 100}%; border-radius: 4px; transition: width 0.3s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.processing_state['status'] == 'error':
        st.error(f"❌ {st.session_state.processing_state['message']}")

def render_sidebar():
    """Render a clean, minimalist sidebar"""
    with st.sidebar:
        # Header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h1 style="font-size: 1.5rem; margin: 0;">🍳 CulinaryCraft</h1>
            <p style="font-size: 0.8rem; color: #666; margin: 0.5rem 0;">AI Recipe Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicators
        ollama_status = "🟢" if check_ollama_connection(st.session_state.ollama_url) else "🔴"
        kb_status = "🟢" if st.session_state.vector_store else "🔴"
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem; padding: 0.5rem; background-color: #f8f9fa; border-radius: 8px;">
            <span>Ollama {ollama_status}</span>
            <span>Knowledge Base {kb_status}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Selection
        if not st.session_state.ollama_models:
            st.session_state.ollama_models = fetch_ollama_models(st.session_state.ollama_url) or Config.DEFAULT_OLLAMA_MODELS
        
        st.session_state.selected_model = st.selectbox(
            "Model",
            st.session_state.ollama_models,
            index=0 if not st.session_state.selected_model else st.session_state.ollama_models.index(st.session_state.selected_model) if st.session_state.selected_model in st.session_state.ollama_models else 0
        )
        
        st.markdown("---")
        
        # Knowledge Base
        st.subheader("📚 Knowledge Base")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=Config.SUPPORTED_FILE_TYPES,
            label_visibility="collapsed",
            help="Upload PDF, TXT, or MD files to enrich your culinary knowledge",
            key="kb_file_uploader"
        )
        
        if uploaded_files:
            if st.button("📥 Process", use_container_width=True, key="process_kb_files"):
                with st.spinner("Processing..."):
                    documents = []
                    for file in uploaded_files:
                        suffix = Path(file.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            tmp_file.write(file.read())
                            tmp_path = tmp_file.name
                        
                        try:
                            if file.name.endswith(".pdf"):
                                loader = PyPDFLoader(tmp_path)
                            else:
                                loader = TextLoader(tmp_path)
                            
                            docs = loader.load()
                            documents.extend(docs)
                        except Exception:
                            pass
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                    
                    if documents:
                        st.session_state.vector_store = initialize_rag_system(
                            documents,
                            st.session_state.embedding_model
                        )
                        if st.session_state.vector_store:
                            st.success(f"✅ {len(documents)} files processed")
                            st.rerun()
        
        if st.session_state.vector_store:
            st.caption(f"📊 {st.session_state.vector_store.index.ntotal} chunks indexed")
        
        st.markdown("---")
        
        # Session Management
        st.subheader("📁 Sessions")
        
        # Create new session
        if st.button("➕ New Session", use_container_width=True, key="new_session"):
            session_id = generate_id()
            new_session = Session(
                id=session_id,
                name=f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                recipes=[]
            )
            save_session(new_session)
            st.session_state.current_session = new_session
            st.rerun()
        
        # List sessions
        sessions = get_all_sessions()
        if sessions:
            for session in sessions[:5]:  # Show last 5 sessions
                is_current = st.session_state.current_session and st.session_state.current_session.id == session['id']
                button_label = f"{'🔵 ' if is_current else ''}{session['name']}"
                
                if st.button(button_label, use_container_width=True, key=f"session_{session['id']}"):
                    loaded_session = load_session(session['id'])
                    if loaded_session:
                        st.session_state.current_session = loaded_session
                        st.rerun()
        
        # Search sessions
        search_query = st.text_input("Search sessions...", key="session_search")
        if search_query:
            search_results = search_sessions(search_query)
            if search_results:
                st.write("Search Results:")
                for result in search_results:
                    if st.button(result['name'], use_container_width=True, key=f"search_{result['id']}"):
                        loaded_session = load_session(result['id'])
                        if loaded_session:
                            st.session_state.current_session = loaded_session
                            st.rerun()

def render_workspace():
    """Render the main workspace"""
    # Show processing indicator at the top
    render_processing_indicator()
    
    # Header
    st.markdown("### 🎯 Recipe Workspace")
    
    # Current session info
    if st.session_state.current_session:
        st.markdown(f"**Current Session:** {st.session_state.current_session.name}")
        st.markdown(f"**Recipes:** {len(st.session_state.current_session.recipes)}")
    else:
        st.info("Please create or select a session from the sidebar")
        return
    
    # Recipe generation
    st.markdown("#### Generate Recipe")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        prompt = st.text_input("Describe the recipe you want to create...", key="recipe_prompt")
    with col2:
        generate_button = st.button("Generate", use_container_width=True, type="primary", key="generate_recipe")
    
    if generate_button and prompt:
        if not st.session_state.selected_model:
            st.error("Please select a model in the sidebar")
            return
        
        with st.spinner("Generating recipe..."):
            recipe_text = generate_recipe_with_ollama(prompt, st.session_state.selected_model, st.session_state.ollama_url)
            
            if recipe_text:
                recipe = parse_recipe_from_text(recipe_text)
                if recipe:
                    st.session_state.current_session.recipes.append(recipe)
                    save_session(st.session_state.current_session)
                    st.success(f"✅ Recipe '{recipe.name}' created and saved!")
                    st.rerun()
                else:
                    st.error("Failed to parse recipe from generated text")
            else:
                st.error("Failed to generate recipe")
    
    # Display recipes
    st.markdown("#### Your Recipes")
    
    if st.session_state.current_session.recipes:
        for i, recipe in enumerate(st.session_state.current_session.recipes):
            with st.expander(f"📜 {recipe.name}", expanded=False, key=f"recipe_{i}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Description:** {recipe.description}")
                    st.markdown(f"**Prep Time:** {recipe.prep_time} minutes")
                    st.markdown(f"**Cook Time:** {recipe.cook_time} minutes")
                    st.markdown(f"**Servings:** {recipe.servings}")
                
                with col2:
                    if st.button("Delete", use_container_width=True, key=f"delete_{recipe.id}"):
                        st.session_state.current_session.recipes.remove(recipe)
                        save_session(st.session_state.current_session)
                        st.rerun()
                
                st.markdown("**Ingredients:**")
                for ingredient in recipe.ingredients:
                    st.markdown(f"- {ingredient['amount']} {ingredient['name']}")
                
                st.markdown("**Instructions:**")
                for j, instruction in enumerate(recipe.instructions):
                    st.markdown(f"{j+1}. {instruction}")
                
                if recipe.nutritional_info:
                    st.markdown("**Nutritional Information:**")
                    nutrition_text = []
                    if "calories" in recipe.nutritional_info:
                        nutrition_text.append(f"Calories: {recipe.nutritional_info['calories']}")
                    if "protein" in recipe.nutritional_info:
                        nutrition_text.append(f"Protein: {recipe.nutritional_info['protein']}g")
                    if "carbs" in recipe.nutritional_info:
                        nutrition_text.append(f"Carbs: {recipe.nutritional_info['carbs']}g")
                    if "fat" in recipe.nutritional_info:
                        nutrition_text.append(f"Fat: {recipe.nutritional_info['fat']}g")
                    
                    st.markdown(", ".join(nutrition_text))
    else:
        st.info("No recipes yet. Generate your first recipe above!")

def render_knowledge_base():
    """Render the knowledge base interface"""
    st.markdown("### 🔍 Knowledge Base")
    
    if st.session_state.vector_store:
        st.success(f"📚 {st.session_state.vector_store.index.ntotal} chunks indexed")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Search knowledge base...", key="kb_search_query")
        with col2:
            k_results = st.selectbox("Results", [3, 5, 10], index=0, key="kb_results_count")
        
        if query:
            with st.spinner("Searching..."):
                docs = st.session_state.vector_store.similarity_search(query, k=k_results)
                for i, doc in enumerate(docs):
                    with st.expander(f"Result {i+1}", expanded=False, key=f"kb_result_{i}"):
                        st.markdown(doc.page_content)
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
    else:
        st.warning("No knowledge base. Upload documents in the sidebar.")
        
        # Show helpful tips
        st.markdown("""
        ### 💡 Tips for Knowledge Base
        - Upload cookbooks, recipe collections, or culinary guides
        - Supported formats: PDF, TXT, MD
        - The AI will use this information to create more relevant recipes
        """)

# Main Application
def main():
    # Configure page
    st.set_page_config(
        page_title="CulinaryCraft AI",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Minimalist CSS
    st.markdown("""
    <style>
        .main-header {
            display: none;
        }
        .stSelectbox > div > div > select {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
        }
        .stTextInput > div > div > input {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
        }
        .processing-spinner {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Check Ollama connection at startup
    if not st.session_state.ollama_checked:
        if check_ollama_connection(st.session_state.ollama_url):
            st.session_state.ollama_models = fetch_ollama_models(st.session_state.ollama_url)
        st.session_state.ollama_checked = True
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🎯 Workspace", "🔍 Knowledge Base"])
    
    with tab1:
        render_workspace()
    
    with tab2:
        render_knowledge_base()
    
    # Footer
    st.markdown("---")
    st.markdown("**CulinaryCraft AI** - Powered by Ollama & FAISS")

if __name__ == "__main__":
    main()