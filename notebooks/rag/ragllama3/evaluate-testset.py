import sqlite3
import pandas as pd
import re

# LLM
from langchain_ollama import ChatOllama

# RAG agent
from rag import run_agentic_rag

# evaluation
from giskard.rag import evaluate
from giskard.rag import KnowledgeBase, QATestset

# to chunk the text
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Client
import giskard
from openai import OpenAI
from giskard.llm.client.openai import OpenAIClient
from giskard.llm.embeddings.openai import OpenAIEmbedding
from giskard.llm.embeddings import set_default_embedding

local_llm = "llama3.1:70b"
llm = ChatOllama(model=local_llm, temperature=0)

# The RAG agent
answer = run_agentic_rag("What is the role of enviromental factors in the prevalence of diarrheal pathogens?")

# Load the generated test set
testset = QATestset.load(".jsonl")

# Define the knowlegde base


# Evaluation
report = evaluate(str(answer),
                testset=testset,
                knowledge_base=knowledge_base,
                metrics=[ragas_context_recall, ragas_context_precision])