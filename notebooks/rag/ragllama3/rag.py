""" This is a RAG agent for MineDD"""

# ---- standard python libraries ---- #
import argparse
import json
import operator
import os
from os.path import exists
from typing_extensions import TypedDict
from typing import List, Annotated
# ---- SQL ---- #
import sqlite3
# ---- langchain ----- #
## - LLM
from langchain_ollama import ChatOllama
from langchain.schema import Document, AIMessage
## - to chunk the text
from langchain.text_splitter import RecursiveCharacterTextSplitter
## - to make/store embeddings
from langchain_community.vectorstores import SKLearnVectorStore
#from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
## - to build/run a langgraph
from langgraph.graph import StateGraph, START, END

local_llm="llama3.2:3b-instruct-fp16"
db='../data/literature_relevant.db'
embeddings_path=f"../embeddings/" + "union.bson"

llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
database_path = db
persist_path = embeddings_path

# state class
# this is the class state
class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question : str # User question
    generation : str # LLM generation
    max_retries : int # Max number of retries for answer generation
    answers : int # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents : List[str] # List of retrieved documents

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_full_text_content(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Retrieve all rows/papers from the table
    cursor.execute(f"SELECT fulltext FROM literature_fulltext;")
    rows = cursor.fetchall()

    # Iterate through the rows (which are papers) and extract text content
    text_content = [row[0] for row in rows if isinstance(row[0], str) and row[0] is not None]

    conn.close()

    return text_content

def text_splitter(database_path):
    docs = extract_full_text_content(database_path)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=5000, chunk_overlap=0
    )

    documents = text_splitter.create_documents([text for text in docs])
    return documents
def create_retriever(persist_path, database_path, documents=None):
    """This function creates/loads embeddings from documents and it returns a retriever"""
    folder = os.path.dirname(persist_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    if exists(persist_path):

        vectorstore = SKLearnVectorStore(
            #embedding=SpacyEmbeddings(model_name='en_core_web_sm'),
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
            persist_path=persist_path,
            serializer="bson")
        print("Vector store was loaded from", persist_path)

    else:
        if documents is None:
            documents=text_splitter(database_path)
        vectorstore = SKLearnVectorStore.from_documents(
            documents=documents,
            #embedding=SpacyEmbeddings(model_name='en_core_web_sm'),
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
            persist_path=persist_path,
            serializer="bson")
        vectorstore.persist()
        print("Vector store was persisted to", persist_path)

    retriever = vectorstore.as_retriever()
    return retriever

# ----------- NODES AND EDGES ------------- #
### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    return {"documents": documents}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    rag_prompt = """You are an assistant for question-answering tasks.

    Here is the context to use to answer the question:

    {context}

    Think carefully about the above context.

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context.

    Use 10 sentences maximum and keep the answer concise.

    Answer:"""

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag AND THAT'S IT

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.

    This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

    Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

    doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question=question)
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)['binary_score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search}


### Edges

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(documents=format_docs(documents),
                                                                               generation=generation.content)
    result = llm_json_mode.invoke([SystemMessage(content=hallucination_grader_instructions)] + [
        HumanMessage(content=hallucination_grader_prompt_formatted)])
    grade = json.loads(result.content)['binary_score']

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(question=question, generation=generation.content)
        result = llm_json_mode.invoke([SystemMessage(content=answer_grader_instructions)] + [
            HumanMessage(content=answer_grader_prompt_formatted)])
        grade = json.loads(result.content)['binary_score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"

retriever = create_retriever(persist_path, database_path)

# --------- Create the StateGraph --------- #
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

# Build graph
# Directly set "retrieve" as the entry point, skipping route_question and web_search
workflow.set_entry_point("retrieve")

# Set edges to transition between nodes
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "generate")  # After grading, generate the response
workflow.add_edge("generate", END)
graph = workflow.compile()
# Compile

def run_agentic_rag(question):
    state = {"question": question, "max_retries": 0}

    documents_list = []

    for event in graph.stream(state, stream_mode="values"):
        if "generation" in event and isinstance(event["generation"], AIMessage):
            # Print only the content of the AIMessage
            print(event["generation"].content)

            # event contains documents (which are stored under the key 'documents')
        if "documents" in event:
            # append the documents to the documents_list, but do not print them
            documents_list.extend(event["documents"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agentic RAG with a custom question.")
    parser.add_argument("question", type=str, help="The question to ask the RAG system.")

    # Parse the argument
    args = parser.parse_args()

    # Pass the question to the run_agentic_rag function
    run_agentic_rag(args.question)



