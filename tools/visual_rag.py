# pip install streamlit==1.46.0
# python -m streamlit run visual_minedd.py

import streamlit as st
import pandas as pd
import os
import json
import time
import faiss
from minedd.rag import PersistentFAISS, SimpleRAG, get_documents_from_directory
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
import dotenv
dotenv.load_dotenv('../notebooks/.env')  # Load environment variables from .env file

# Important paths and configurations
PAPERS_DIRECTORY = Path.home() / "papers_campylo/" # Directory containing the PDF papers
EMBEDDING = "mxbai-embed-large:latest" # Embedder model to use
VECTOR_STORE_PATH = "../minedd/outputs/minedd_campylo_index"

# Define available models
AVAILABLE_MODELS = [
    "ollama/llama3.2:latest",
    "google_genai/gemini-2.5-flash-lite-preview-06-17", 
]


# Initialize the Query engine with selected model
@st.cache_resource
def initialize_engine(selected_model):

    model_provider, model_name = selected_model.split("/")
    #Generative Model
    llm = init_chat_model(model_name, 
                          model_provider=model_provider,
                            temperature=0.0,
                            max_retries=2,
                            max_output_tokens=8000
                          )
    # Embeddings model
    embeddings = OllamaEmbeddings(model=EMBEDDING)

    # Index and Store Embeddings
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store_path = VECTOR_STORE_PATH
    vector_store = PersistentFAISS(
        index_path=vector_store_path,
        index=index, 
        embeddings_engine=embeddings,
        include_bm25=True
        )
    
    # Create RAG Engine
    rag_engine = SimpleRAG(
        embeddings=embeddings,
        generative_llm=llm,
        vector_store=vector_store
    )

    # ## Load Docs + Create a Document object for each chunk
    if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
        print(f"Loading existing vector store from {vector_store_path}")
        vector_store.initialize()
    else:
        print(f"No existing vector store found at {vector_store_path}, Chunking and Loading Documents to create one...")
        docs = get_documents_from_directory(
            directory=PAPERS_DIRECTORY,
            extensions=['.json'],
            chunk_size=8, # Number of sentences to merge into one Document
            overlap=2 # Number of sentences to overlap between chunks
        )
        vector_store.initialize(documents=docs)

    return rag_engine


# Function to get document names
@st.cache_data
def get_documents():
    documents = {}
    if os.path.exists(PAPERS_DIRECTORY):
        for filename in os.listdir(PAPERS_DIRECTORY):
            if filename.endswith(('.pdf', '.md')):
                documents[filename] = get_document_parsed_content(filename)
    return documents


@st.cache_data
def get_document_parsed_content(filename):
    doc_json_path = PAPERS_DIRECTORY / f"{filename[:-4]}.json"
    doc_content = {}
    if os.path.exists(doc_json_path):
        doc_content = json.load(open(doc_json_path, 'r', encoding='utf-8'))
    else:
        print("Problem loading document content from", doc_json_path)
    return doc_content




# Streamlit app
st.title('Paper Querier - Simple RAG Edition')
st.write('Ask questions about your document collection')

# Model Selection
st.subheader('⚙️ Configuration')
selected_model = st.selectbox(
    'Choose Model:',
    AVAILABLE_MODELS,
    index=0,
    help='Select the AI model to use for answering questions'
)

# Initialize engine with selected model
model_locale = "[LOCAL]" if "ollama" in selected_model else "[REMOTE]"
if 'current_model' not in st.session_state or st.session_state.current_model != selected_model:
    st.session_state.current_model = selected_model
    st.session_state.engine = initialize_engine(selected_model)
    st.success(f'Model switched to: **{selected_model}**  {model_locale}')


# Available Documents Section
st.subheader('📄 Available Documents')
with st.expander("View Available Documents", expanded=False):
    documents = get_documents()
    st.write(len(documents))
    try:
        show_docs_df = pd.DataFrame([
            (documents[k].get('doc_key', 'NoKey'),
             documents[k].get('title', ", ".join(documents[k].keys())), 
             k) for k in documents.keys()], columns=['DocKey', 'Title', 'Document Filename'])
    except KeyError:
        show_docs_df = pd.DataFrame(columns=['Title', 'Document Filename'])
        st.error("Error loading document titles. Please check the document structure.")

    
    if documents:
        st.write(f"**Total Documents:** {len(documents)}")
        st.dataframe(show_docs_df, use_container_width=True)
        # # Display documents in a nice format
        # for i, doc_name in enumerate(documents, 1):
        #     st.write(f"**{i}.** {doc_name}")
    else:
        st.info("No documents available")

# Add some spacing
st.markdown("---")


# Create columns for buttons
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    # Text input for the question
    question = st.text_input('Enter your question:', value='What are the climatic associated factors of Campylobacter in nordic countries?')

with col2:
    search_button = st.button('🔍 Search', type='primary')

with col3:
    reset_button = st.button('🔄 Reset')

col4, col5 = st.columns([4, 4])
with col4:
    answer_length = st.number_input('Answer Length (characters)', min_value=10, max_value=4000, value=400, step=50, key='answer_length')

with col5:
    top_k = st.number_input('Top-K Contexts', min_value=1, max_value=20, value=10, step=1, key='top_k')
    

# Handle reset functionality
if reset_button:
    st.rerun()

# Handle search functionality
if search_button and question:
    with st.spinner('Searching for answers...'):
        try:
            start_time = time.time()
            contexts, response = st.session_state.engine.query(question, 
                                                               k=top_k, 
                                                               answer_length=answer_length, 
                                                               hybrid=True, 
                                                               num_candidates=top_k*2)
            execution_time = time.time() - start_time
            
            # Display results in a nicely formatted way
            st.success(f'Search completed in {execution_time:.2f} seconds!')
            
            # Answer section
            st.subheader('💡 Answer')
            st.markdown(response)
            
            # # # Sources section
            # shown_citations = set()
            # if result['citations']:
            #     st.subheader('📚 Sources')
            #     for i, citation in enumerate(result['citations']):
            #         if citation not in shown_citations:
            #             st.write(f"**{i+1}.** {citation}")
            #             shown_citations.add(citation)
            
            # Original Contexts section (if available). But this is not nice because PyPDF does not parse well the contexts so they are crappy
            if contexts:
                st.markdown("---")
                st.header('🔗 Relevant Contexts')
                for i, langchain_doc in enumerate(contexts):
                    st.write(f"### {i+1}) {langchain_doc.metadata['title']}\n#### {langchain_doc.metadata['section']}. Pages: {langchain_doc.metadata['pages']}")
                    st.write(f"* **Content**: {langchain_doc.page_content}")
                    # for k, v in langchain_doc.metadata.items():
                    #     if k not in ["title", "section"]:
                    #         st.write(f"* **{k.title()}**: {v}")
                    
        except Exception as e:
            st.error(f'An error occurred: {str(e)}')
            
elif search_button and not question:
    st.warning('Please enter a question before searching.')

# Add some styling and information
st.markdown('---')
st.markdown('*Enter your question above and click Search to get answers from your document collection.*')
