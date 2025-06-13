# Source: https://python.langchain.com/docs/tutorials/rag/
# %pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph 
# pip install -qU "langchain[cohere]"
# pip install -qU langchain-huggingface
# pip install faiss-cpu
# pip install cohere

import os
import cohere
import faiss
import numpy as np
import pandas as pd
from minedd.document import DocumentMarkdown
from dotenv import load_dotenv
load_dotenv(override=True)


cohere_api_key = os.getenv('COHERE_API_KEY')
co = cohere.Client(cohere_api_key)

# llm = init_chat_model("command-r-plus", model_provider="cohere")

# # Embeddings model
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# embedding_dim = len(embeddings.embed_query("hello world"))


# vector_store = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )




# # Create a Document object for each chunk
# docs = []
# for i, chunk in enumerate(chunks):
#     doc = Document(page_content=chunk, metadata={"source": "paper_text.md", "title": markdown_paper.get_title(), "chunk_index": i})
#     docs.append(doc)

# # Index chunks
# vector_store.add_documents(documents=docs)

def create_document_database(filenames):
    chunks = []
    # Load Document(s)
    for filename in filenames:
        markdown_paper = DocumentMarkdown(md_path=filename)
        chunks += markdown_paper.convert_to_chunks(mode="chars", chunk_size=1500, overlap=100)
    # Index
    print(len(chunks))
    response = co.embed(
    texts=chunks,
    input_type="search_document",
    ).embeddings

    embeds = np.array(response)
    embedding_dim = embeds.shape[1]
    print(embeds.shape)
    # Create Index (and Vector Store?)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.float32(embeds))
    return index, chunks



def search(query, texts_np, index, number_of_results=3):

    # 1. Get the query's embedding
    query_embed = co.embed(texts=[query],
                input_type="search_query",).embeddings[0]

    # 2. Retrieve the nearest neighbors
    distances , similar_item_ids = index.search(np.float32([query_embed]), number_of_results)

    # 3. Format the results
    results = pd.DataFrame(data={'texts': texts_np[similar_item_ids[0]],
                                'distance': distances[0]})

    # 4. Print and return the results
    print(f"Query:'{query}'\nNearest neighbors:")
    return results



## Load Docs
filenames = ["notebooks/outputs/paper_text.md"]
index, texts = create_document_database(filenames)
texts_np = np.array(texts) # Convert texts list to numpy for easier indexing


query = "How do climate variables affect the spread of rotavirus?"

# 1- Retrieval
# We'll use embedding search. But ideally we'd do hybrid
results = search(query, texts_np, index)

# 2- Grounded Generation
docs_dict = [{'text': text} for text in results['texts']]
response = co.chat(
    message = query,
    documents=docs_dict
)

print(response.text)