# https://python.langchain.com/docs/integrations/graphs/memgraph/
import networkx as nx
from langchain.chat_models import init_chat_model
from langchain_memgraph.chains.graph_qa import MemgraphQAChain
from langchain_memgraph.graphs.memgraph import MemgraphLangChain # pip install langchain-memgraph
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
import os
import networkx as nx
from pyvis.network import Network
# import matplotlib.pyplot as plt


class GraphRAGSystem:
    def __init__(self, llm):
        """Initialize the GraphRAG system with Memgraph connection"""
        self.graph = MemgraphLangChain(
            url="bolt://localhost:7687", 
            username="", 
            password="", 
            refresh_schema=False
        )
        self.llm = llm
        self.graph_transformer = LLMGraphTransformer(llm=llm)
        self.qa_chain = MemgraphQAChain.from_llm(
            self.llm,
            graph=self.graph,
            model_name="llama3.2:latest",
            top_k=3,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            verbose=False
            )
        
    def setup_sample_data(self):
        """Create sample knowledge graph data"""
        # Drop graph
        self.graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
        self.graph.query("DROP GRAPH")
        self.graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")
        # Clear existing data
        self.graph.query("MATCH (n) DETACH DELETE n")
        # Insert sample data about companies and technologies
        sample_data = """
        CREATE
        (openai:Company {name: 'OpenAI', founded: '2015', type: 'AI Research'}),
        (microsoft:Company {name: 'Microsoft', founded: '1975', type: 'Technology'}),
        (langchain:Framework {name: 'LangChain', type: 'AI Framework', language: 'Python'}),
        (memgraph:Database {name: 'Memgraph', type: 'Graph Database', query_language: 'Cypher'}),
        (gpt:Model {name: 'GPT-4', type: 'Language Model', parameters: '1.76T'}),
        (rag:Technique {name: 'RAG', full_name: 'Retrieval Augmented Generation', type: 'AI Technique'}),
        (graphrag:Technique {name: 'GraphRAG', type: 'Enhanced RAG', uses_graphs: true}),
        
        (openai)-[:DEVELOPED]->(gpt),
        (microsoft)-[:INVESTED_IN]->(openai),
        (langchain)-[:SUPPORTS]->(memgraph),
        (langchain)-[:IMPLEMENTS]->(rag),
        (graphrag)-[:EXTENDS]->(rag),
        (graphrag)-[:USES]->(memgraph),
        (memgraph)-[:STORES]->(langchain)
        """
        
        self.graph.query(sample_data)
        self.graph.refresh_schema()
        print("Sample data inserted successfully!")
        
    def create_from_documents(self, documents):
        """Create knowledge graph from unstructured text documents"""
        # Transform documents into graph format
        graph_documents = self.graph_transformer.convert_to_graph_documents(documents)
        # Empty the database
        self.graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
        self.graph.query("DROP GRAPH")
        self.graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")
        # Create KG
        self.graph.add_graph_documents(graph_documents)
        self.graph.refresh_schema()
        print(f"Knowledge graph created from {len(documents)} documents!")
    
    def add_documents(self, documents):
        """Add new documents to the existing knowledge graph"""
        # Transform documents into graph format
        graph_documents = self.llm_graph_transformer.convert_to_graph_documents(documents)
        # Add to the graph
        self.graph.add_graph_documents(graph_documents)
        self.graph.refresh_schema()
        print(f"Added {len(documents)} new documents to the knowledge graph!")

    def get_schema(self):
        """Get the current graph schema"""
        return self.graph.get_schema
        
    def direct_cypher_query(self, cypher_query):
        """Execute a direct Cypher query"""
        return self.graph.query(cypher_query)

    def query(self, question):
        """Query the knowledge graph with natural language"""            
        result = self.qa_chain.invoke(question)
        return result
    
    def visualize_graph(self, nodes=None, edges=None, html_path=None):
        """Visualize the knowledge graph using NetworkX
            It returns the NetworkX graph object and optionally saves it as an HTML file.
        """
        # Get relevant nodes and relationships (default: Get All)
        nodes = nodes else self.graph.query("MATCH (n) RETURN n")
        edges = edges else self.graph.query("MATCH (n)-[r]->(m) RETURN n, r, m")

        G = nx.Graph()
        
        # Add nodes
        try:
            for node in nodes:
                # print(f"Node {node}")
                G.add_node(node['n']['id'], **node['n'])
        except Exception as e:
            print(f"Error adding nodes: {e}\nNode Example: {nodes[0]}")
        
        # Add edges
        try:
            for edge in edges:
                # print(f"Edge {edge}")
                G.add_edge(edge['n']['id'], edge['m']['id'], edge_label=edge['r'][1])
        except Exception as e:
            print(f"Error adding edges: {e}\nEdge Example: {edges[0]}")
        
        # Convert edge_label to both label (visible) and title (hover)
        for source, target, data in G.edges(data=True):
            if 'edge_label' in data:
                # data['label'] = str(data['edge_label'])  # Permanently visible
                data['title'] = f"Relationship: {data['edge_label']}"  # Hover tooltip

        # Use Pyvis to create an HTML interactive graph
        if html_path:
            if not os.path.exists(html_path):
                os.makedirs(os.path.dirname(html_path), exist_ok=True)
            net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
            net.from_nx(G)
            net.save_graph(html_path)

        # FOR LATER: Display in Streamlit
        # with open("interactive_graph.html", 'r', encoding='utf-8') as f:
        #     html_content = f.read()
        # components.html(html_content, height=500)
        
        return G


def save_graph_to_networkx_file(graph, filename="knowledge_graph.gexf"):
    """Save the graph to a GEXF file"""
    nx.write_gexf(graph, filename)
    print(f"Graph saved to {filename}")

def load_graph_from_networkx_file(filename="knowledge_graph.gexf"):
    """Load a graph from a GEXF file"""
    return nx.read_gexf(filename)

# Usage example
def main(llm):
    # Initialize the GraphRAG system
    print("Initializing GraphRAG system...")
    graph_rag = GraphRAGSystem(llm)
    
    # Set up sample data
    print("Setting up sample data...")
    graph_rag.setup_sample_data()
    
    # Display schema
    print("\nGraph Schema:")
    print(graph_rag.get_schema())
    
    # Example queries
    questions = [
        "What companies are involved in AI development?",
        "What frameworks work with graph databases?",
        "How is GraphRAG related to traditional RAG?",
        "What database stores LangChain data?"
    ]
    
    print("\n" + "="*50)
    print("QUERYING THE KNOWLEDGE GRAPH")
    print("="*50)
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        try:
            result = graph_rag.query(question)
            print(result)
            print(f"Answer: {result['result']}")
            if 'intermediate_steps' in result:
                print(f"Generated Cypher: {result['intermediate_steps'][0]['query']}")
        except Exception as e:
            print(f"Error: {e}")

def build_knowledge_graph(llm):
    text = """
        Charles Robert Darwin was an English naturalist, geologist, and biologist,
        widely known for his contributions to evolutionary biology. His proposition that
        all species of life have descended from a common ancestor is now generally
        accepted and considered a fundamental scientific concept. In a joint
        publication with Alfred Russel Wallace, he introduced his scientific theory that
        this branching pattern of evolution resulted from a process he called natural
        selection, in which the struggle for existence has a similar effect to the
        artificial selection involved in selective breeding. Darwin has been
        described as one of the most influential figures in human history and was
        honoured by burial in Westminster Abbey.
    """
    documents = [Document(page_content=text)]

    # Initialize the GraphRAG system
    print("Initializing GraphRAG system...")
    graph_rag = GraphRAGSystem(llm)
    graph_rag.create_from_documents(documents)
    graph_rag.visualize_graph()

if __name__ == "__main__":
    llm = init_chat_model(
            model="llama3.2:latest",
            model_provider="ollama",
            temperature=0.0,
            )
    # main(llm)
    build_knowledge_graph(llm)
