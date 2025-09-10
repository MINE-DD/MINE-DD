# pip install streamlit==1.46.0
# python -m streamlit run visual_minedd.py

import streamlit as st
import pandas as pd
import os
import json
import time
import re
from paperqa.settings import Settings, AgentSettings, ParsingSettings
from minedd.embeddings import Embeddings
from minedd.query import Query
import dotenv
dotenv.load_dotenv('minedd/.env')  # Load environment variables from .env file

# Important paths and configurations
PAPERS_DIRECTORY = os.getenv("PAPERS_DIRECTORY", None) # Directory containing the PDF papers
EMBEDDING = os.getenv("LLM_EMBEDDER", "mxbai-embed-large:latest") # Embedder model to use
EMBEDDINGS_DIR = os.getenv("MINEDD_EMBEDDINGS_PATH", "minedd-embeddings.pkl") # Pickled file containing the embeddings generated with minedd (paperQA)
OUTPUTS_DIRECTORY = 'outputs'  # Directory for output files

# Define available models
AVAILABLE_MODELS = [
    "ollama/llama3.2:latest",
    "gemini/gemini-2.5-flash-lite-preview-06-17", 
]

if not PAPERS_DIRECTORY or not os.path.exists(PAPERS_DIRECTORY):
    st.error(f"The provided Paper Directory '{PAPERS_DIRECTORY}' does not exist!")
    exit() 

# Initialize the Query engine with selected model
@st.cache_resource
def initialize_engine(selected_model):

    # If the PaperQA inted does not exist we have to create it...
    created_embeddings = None
    if EMBEDDINGS_DIR and not os.path.exists(EMBEDDINGS_DIR):
        print(f"PaperQA Embeddings index {EMBEDDINGS_DIR} does not exist. Loading papers and creating a new one...")
        created_embeddings = Embeddings(
            model="ollama/llama3.2:latest", # this param is irrelevant at this step anyway...
            embedding_model=f"ollama/{EMBEDDING}",
            paper_directory=PAPERS_DIRECTORY, # type: ignore
            output_embeddings_path=EMBEDDINGS_DIR,
        )
        pdf_file_list = created_embeddings.prepare_papers()
        print(f"Found {len(pdf_file_list)} papers in the directory. Creating embeddings...")
        created_embeddings.process_papers(pdf_file_list)
        print(f"Embeddings created and saved to {EMBEDDINGS_DIR}")
        print("CREATED", created_embeddings)
        del created_embeddings

    local_llm_config = {
        "model_list": [
            {
                "model_name": selected_model,
                "litellm_params": {
                    "model": selected_model,
                    # Uncomment if using a local server
                    # "api_base": "http://0.0.0.0:11434",
                },
                "answer": {
                    "evidence_k": 15,
                    "evidence_detailed_citations": True,
                    "evidence_summary_length": "about 300 words",
                    "answer_max_sources": 5,
                    "answer_length": "about 600 words, but can be longer",
                    "max_concurrent_requests": 10,
                    "answer_filter_extra_background": False
                },
                "parsing": {
                    "use_doc_details": True
                },
                "prompts" : {"use_json": False}
            }
        ]
    }

    query_settings = Settings(
        llm=selected_model,
        llm_config=local_llm_config,
        summary_llm=selected_model,
        summary_llm_config=local_llm_config,
        paper_directory=PAPERS_DIRECTORY, # type: ignore
        embedding=f"ollama/{EMBEDDING}",
        agent=AgentSettings(
            agent_llm=selected_model,
            agent_llm_config=local_llm_config,
            return_paper_metadata=True
        ),
        parsing=ParsingSettings(
            chunk_size=2500,
            overlap=250
        ),
        prompts={"use_json": False} # type: ignore
    )

    engine = Query(
        model=selected_model,
        paper_directory=PAPERS_DIRECTORY, # type: ignore
        output_dir=OUTPUTS_DIRECTORY,
    )
    engine.settings = query_settings
    engine.load_embeddings(EMBEDDINGS_DIR)
    return engine


# Function to get document names
@st.cache_data
def get_documents():
    documents = {}
    embeddings_db = Embeddings()
    embeddings_db.load_existing_embeddings(EMBEDDINGS_DIR)
    print(embeddings_db)
    if PAPERS_DIRECTORY and os.path.exists(PAPERS_DIRECTORY):
        try:
            docs_df = embeddings_db.get_docs_details()
            print(docs_df)
            docs_df = docs_df[["docname", "title"]]
            filenames_df = pd.DataFrame({"filename": os.listdir(PAPERS_DIRECTORY)})
            filenames_df["title"] = filenames_df["filename"].apply(lambda row: re.sub(" +", " ", row.strip(".pdf").replace("_", " ").replace("-", " ")).strip())
            filenames_df = filenames_df.merge(docs_df, how="left", on="title") # type: ignore
        except Exception as e:
            print(f"Error loading document titles: {str(e)}")
            docs_df = pd.DataFrame([])
        for filename in os.listdir(PAPERS_DIRECTORY):
            if filename.endswith(('.pdf', '.md')):
                documents[filename] = get_document_parsed_content(filename)

    return documents, docs_df

@st.cache_data
def get_document_parsed_content(filename):
    doc_json_path = f"PAPERS_DIRECTORY/{filename.strip('.pdf')}.json"
    doc_content = {}
    if os.path.exists(doc_json_path):
        doc_content = json.load(open(doc_json_path, 'r', encoding='utf-8'))
    return doc_content




# Streamlit app
st.title('Paper Querier - Minedd Edition')
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
        documents, show_docs_df = get_documents()
    except KeyError:
        show_docs_df = pd.DataFrame(columns=['docname', 'title'])
        st.error("Error loading document titles. Please check the document structure.")

    
    if documents:
        st.write(f"**Total Documents:** {len(documents)}")
        st.dataframe(show_docs_df, use_container_width=True)
    else:
        st.info("No documents available")

# Add some spacing
st.markdown("---")



# Create columns for buttons
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    # Text input for the question
    question = st.text_input('Enter your question:', value='What are the climatic associated factors of Cryptosporidium?')

with col2:
    search_button = st.button('🔍 Search', type='primary')

with col3:
    reset_button = st.button('🔄 Reset')


    

# Handle reset functionality
if reset_button:
    st.rerun()

# Handle search functionality
if search_button and question:
    with st.spinner('Searching for answers...'):
        try:
            start_time = time.time()
            result = st.session_state.engine.query_single(question, max_retries=3)
            execution_time = time.time() - start_time
            if result:
                # Display results in a nicely formatted way
                st.success(f'Search completed in {execution_time:.2f} seconds!')
                
                # Answer section
                st.subheader('💡 Answer')
                answer_without_references = result['answer'].split('\nReferences')[0].strip()
                st.markdown(answer_without_references)
            
                # # Sources section
                shown_citations = set()
                if result['citations']:
                    st.subheader('📚 Sources')
                    for i, citation in enumerate(result['citations']):
                        if citation not in shown_citations:
                            st.write(f"**{i+1}.** {citation}")
                            shown_citations.add(citation)
                # Intermediate steps before generating the final summary. But it is too long and even confusing...
                # st.subheader('📝 Model Summaries')
                # st.write(f"CONTEXT: {result['context']}")

                # Original Contexts section (if available). But this is not nice because PyPDF does not parse well the contexts so they are crappy
                raw_response = result.get('raw_response')
                if raw_response:
                    texts = [context.text for context in raw_response.contexts]
                    st.subheader('🔗 Relevant Contexts')
                    for i, text in enumerate(texts):
                        st.write(f"**{i+1}. {text.name})**") # ... {text.text} ...
                    
        except Exception as e:
            st.error(f'An error occurred: {str(e)}')
            
elif search_button and not question:
    st.warning('Please enter a question before searching.')

# Add some styling and information
st.markdown('---')
st.markdown('*Enter your question above and click Search to get answers from your document collection.*')
