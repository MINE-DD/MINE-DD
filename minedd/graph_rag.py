# https://python.langchain.com/docs/integrations/graphs/memgraph/
import networkx as nx
import pickle
import asyncio
import random
from pathlib import Path
from typing import Optional
import matplotlib.colors as mcolors
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_memgraph.chains.graph_qa import MemgraphQAChain
from langchain_memgraph.graphs.memgraph import MemgraphLangChain # pip install langchain-memgraph
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from minedd.document import get_documents_from_directory
from pyvis.network import Network
from dotenv import load_dotenv
load_dotenv(override=True)
# import matplotlib.pyplot as plt


class MineddRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Campylobacter, Latin America, wind speed. "
            "Can be a short noun phrase. Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Pathogen, Location, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted tail entity like Campylobacter, Latin America, wind speed. "
            "Can be a short noun phrase. Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Pathogen, Location, etc"
    )

def create_minedd_prompt(
    node_labels: Optional[list[str]] = None, rel_types: Optional[list[str]] = None
) -> ChatPromptTemplate:
    node_labels_str = str(node_labels) if node_labels else ""
    rel_types_str = str(rel_types) if rel_types else ""
    documents_domain_description = ("The texts are medical papers describing how different environmental variables affect the incidence of pathogens "
                               "in different geographical areas (GeoAreas), being more present or less present than expected in different locations "
                               "that humans interact with (like air, water or surfaces)")
    # node_label_definitions = {
    #     "MetheorologicalVariable": (
    #         "Environmental Variables, Meteorological Variables, Hydrometeorological conditions that can be "
    #         "negatively associated or positively correlated with Pathogens and present in Locations and Geographical areas",
    #     ),
    #     "Pathogen": (),
    #     "Location": (),
    #     "GeoArea": ()
    # }
    node_label_examples = {
        "Metheorological_Variable": [
            "daily total precipitation volume (mm)"
            "daily total surface runoff (mm)"
            "surface pressure (mbar)"
            "wind speed (m/s)"
            "relative humidity (%)"
            "soil moisture (%)"
            "solar radiation (W/m2)"
            "specific humidity (kg/kg)"
            "average daily temperatures"
        ],
        "Pathogen": ["rotavirus", "campylobacter", "cryptosporidium", "giardia"],
        "Location": [
            "soil",
            "solid surfaces",
            "airborne",
            "waterborne"
            "host factors",
            ],
        "Geo_Area": [
            "tropical areas",
            "developing countries",
            "Italy",
            "Ghana",
            "Tanzania",
            "West Africa",
            "Latinamerican countries"
            ]
    }
    relation_examples = [
        ("MetheorologicalVariable", "POS_ASSOCIATED_WITH", "Pathogen"),
        ("MetheorologicalVariable", "NEG_ASSOCIATED_WITH", "Pathogen"),
        ("MetheorologicalVariable", "PRESENT_IN", "Location"),
        ("Pathogen", "PRESENT_IN", "Location"),
        ("Pathogen", "PRESENT_IN", "GeoArea"),
    ]
    examples = [
        {
            "text": (
                "These results suggest that the effect of climate on rotavirus transmission " 
                "was mediated by four independent mechanisms: waterborne dispersal, airborne dispersal, " 
                "virus survival on soil and surfaces, and host factors. "
            ),
            "head": "waterborne dispersal",
            "head_type": "MetheorologicalVariable",
            "relation": "PRESENT_IN",
            "tail": "soil and surfaces, and host factors",
            "tail_type": "Location",
        },
        {
            "text": (
                "In our analysis there was also an overall sta- tistically significant association with precipitation. " 
                "Campylobacter survival is enhanced in wet conditions [31] and the transmission of Campylobacter " 
                "from the environment to humans may be greater during wetter conditions [7,32]"
            ),
            "head": "wet conditions",
            "head_type": "MetheorologicalVariable",
            "relation": "POS_ASSOCIATED_WITH",
            "tail": "Campylobacter",
            "tail_type": "Pathogen",
        },
        {
            "text": (
                "For the related variables of precipitation and surface runoff, the adjusted effect model predicted "
                "significant U-shaped associations in the low extreme, with lowest probability of rotavirus infection "
                "just below the site- specific mean value"
            ),
            "head": "precipitation",
            "head_type": "MetheorologicalVariable",
            "relation": "NEG_ASSOCIATED_WITH",
            "tail": "rotavirus",
            "tail_type": "Pathogen",
        },
        {
            "text": (
                "For the related variables of precipitation and surface runoff, the adjusted effect model predicted "
                "significant U-shaped associations in the low extreme, with lowest probability of rotavirus infection "
                "just below the site- specific mean value"
            ),
            "head": "surface runoff",
            "head_type": "MetheorologicalVariable",
            "relation": "NEG_ASSOCIATED_WITH",
            "tail": "rotavirus",
            "tail_type": "Pathogen",
        },
        {
            "text": "In contrast to the absolute effect, the adjusted models suggest that somewhat wet conditions—high rainfall generating surface runoff—are associated with a higher risk of rotavirus transmission, whereas relatively low rainfall also brings higher risk. ",
            "head": "high rainfall generating surface runoff",
            "head_type": "MetheorologicalVariable",
            "relation": "POS_ASSOCIATED_WITH",
            "tail": "rotavirus",
            "tail_type": "Pathogen",
        },
        {
            "text": "In contrast to the absolute effect, the adjusted models suggest that somewhat wet conditions—high rainfall generating surface runoff—are associated with a higher risk of rotavirus transmission, whereas relatively low rainfall also brings higher risk. ",
            "head": "whereas relatively low rainfall",
            "head_type": "MetheorologicalVariable",
            "relation": "POS_ASSOCIATED_WITH",
            "tail": "rotavirus",
            "tail_type": "Pathogen",
        },
        {
            "text": "Additionally, the incidence of diarrhoeal rotavirus in the site in Tanzania, which had not introduced the vaccine, was notably low. ",
            "head": "diarrhoeal rotavirus",
            "head_type": "Pathogen",
            "relation": "PRESENT_IN",
            "tail": "Tanzania",
            "tail_type": "GeoArea",
        },
    ]

    # TODO: Delete HARDCODES! Pass all the relations and nodes as parameters...
    node_labels = [n for n,_ in node_label_examples.items()]
    rel_types = [r[1] for r in relation_examples]

    node_labels_str = str([f"\nEntity {name} - Examples: {str(examples)}" for name,examples in node_label_examples.items()]) if node_labels else ""
    rel_types_str = str(relation_examples) if rel_types else ""
    
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        f"scientific text. {documents_domain_description}"
        "You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one of the types "
        "from the provided list in the user prompt.",
        f'The "head_type" key must contain the type of the extracted head entity, '
        f"which must be one of the types from {node_labels_str}."
        if node_labels
        else "",
        f'The "relation" key must contain the type of relation between the "head" '
        f'and the "tail", which must be one of the relations from {rel_types_str}.'
        if rel_types
        else "",
        f'The "tail" key must represent the text of an extracted entity which is '
        f'the tail of the relation, and the "tail_type" key must contain the type '
        f"of the tail entity from {node_labels_str}."
        if node_labels
        else "",
        "Attempt to extract as many entities and relations as you can." 
        "Remember each pice of texts can have more than one triple. Maintain "
        "Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=MineddRelation)

    human_string_parts = [
        "Based on the following example, extract entities and "
        "relations from the provided text.\n\n",
        "Use the following entity types, don't use other entity "
        "that is not defined below:"
        "# ENTITY TYPES:"
        "{node_labels}"
        if node_labels
        else "",
        "Use the following relation types, don't use other relation "
        "that is not defined below:"
        "# RELATION TYPES:"
        "{rel_types}"
        if rel_types
        else "",
        "Below are a number of examples of text and their extracted "
        "entities and relationships."
        "{examples}\n"
        "For the following text, extract entities and relations as "
        "in the provided example."
        "{format_instructions}\nText: {input}",
    ]
    human_prompt_string = "\n".join(filter(None, human_string_parts))
    human_prompt = PromptTemplate(
        template=human_prompt_string,
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": node_labels,
            "rel_types": rel_types,
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt

class GraphRAGDefault:
    def __init__(self, llm, database_name="database", graph_cache_file="my_kg.pkl"):
        """Initialize the GraphRAG system with Memgraph connection"""
        self.graph = MemgraphLangChain(
            url="bolt://localhost:7687", 
            username="", 
            password="", 
            database=database_name,
            refresh_schema=False
        )
        self.llm = llm
        self.graph_cache_file = graph_cache_file
        self.graph_documents: list[GraphDocument] = []
        self.graph_transformer = LLMGraphTransformer(
            llm=llm,
            prompt=create_minedd_prompt(),
            #vallowed_nodes=["PERSON", "PUBLICATION", "LOCATION", "NATIONALITY", "STATEMENT"]
            )
        self.qa_chain = MemgraphQAChain.from_llm(
            self.llm,
            graph=self.graph,
            model_name="llama3.2:latest",
            top_k=3,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            verbose=False
            )
    
    def save_graph_docs(self):
        """Save graph documents using pickle for fast loading."""
        if self.graph_cache_file:
            with open(self.graph_cache_file, 'wb') as f:
                pickle.dump(self.graph_documents, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_graph_docs(self):
        """Load graph documents from pickle file."""
        if self.graph_cache_file:
            with open(self.graph_cache_file, 'rb') as f:
                self.graph_documents = pickle.load(f)
        self.graph.refresh_schema()
        print(f"Loaded {len(self.graph_documents)} GraphDocuments from {self.graph_cache_file}...")
        
    async def create_from_documents(self, documents):
        """Create knowledge graph from unstructured text documents"""
        # Transform documents into graph format
        self.graph_documents = await self.graph_transformer.aconvert_to_graph_documents(documents)
        # Empty the database
        self.graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
        self.graph.query("DROP GRAPH")
        self.graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")
        # Create KG
        self.graph.add_graph_documents(self.graph_documents, include_source=True)
        self.graph.refresh_schema()
        self.save_graph_docs()
        print(f"Knowledge graph created using from {len(self.graph_documents)} documents!")
    
    def add_documents(self, documents):
        """Add new documents to the existing knowledge graph"""
        # if len(documents) == 0:
        #     print("No documents to add.")
        #     return
        # When adding documents, we assume the graph is already initialized
        raise NotImplementedError("Adding documents is not implemented yet. Please use 'create_from_documents' to initialize the graph from scratch.")

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
    
    def visualize_graph(self, nodes_with_labels=None, edges=None, html_path=None):
        """Visualize the knowledge graph using NetworkX
            It returns the NetworkX graph object and optionally saves it as an HTML file.
        """
        # Get relevant nodes and relationships (default: Get All)
        if nodes_with_labels is None:
            nodes_with_labels = self.graph.query("MATCH (n) RETURN n, labels(n) as node_types")
        if edges is None:
            edges = self.graph.query("MATCH (n)-[r]->(m) RETURN n, r, m")

        G = nx.Graph()
        
        ### Add nodes
        try:
            for result in nodes_with_labels:
                node = result['n']
                node_types = result['node_types']
                node['type'] = '+'.join(node_types)
                print(">>> ",node)
                node_id = node['id']
                G.add_node(node_id, **node)
        except Exception as e:
            print(f"Error adding nodes: {e}\nNode Example: {nodes_with_labels[0]}")
        
        ### Add edges
        try:
            for edge in edges:
                # print(f"Edge {edge}")
                G.add_edge(edge['n']['id'], edge['m']['id'], edge_label=edge['r'][1])
        except Exception as e:
            print(f"Error adding edges: {e}\nEdge Example: {edges[0]}")

        ### Colorize the Graph
        # Extract all unique node types from the graph
        node_types = set()
        for node_id, node_data in G.nodes(data=True):
            if 'type' in node_data:
                node_types.add(node_data['type'])
        # Generate a color palette
        colors = generate_color_palette(len(node_types))
        type_color_map = dict(zip(node_types, colors))

        ### Add Node Properties to Display in Tooltip
        for node_id, node_data in G.nodes(data=True):
            # Apply Color to NodeTypes
            if 'type' in node_data:
                G.nodes[node_id]['color'] = type_color_map[node_data['type']]
            else:
                G.nodes[node_id]['color'] = "#187286"  # Default color for nodes without type
            # Create tooltip content
            tooltip_parts = []
            tooltip_parts.append(f"ID: {node_id}")
            if "content" in node_data:
                tooltip_parts.append(f"CONTENT: {node_data['content'][:100]}...")
            # Add other properties
            for key, value in node_data.items():
                if key not in ["id", "content", "title", "color"]:
                    tooltip_parts.append(f"{key.title().upper()}: {value}")
            # Set the title attribute (this becomes the tooltip)
            G.nodes[node_id]["title"] = "\n".join(tooltip_parts)
        
        # Convert edge_label to both label (visible) and title (hover)
        for source, target, data in G.edges(data=True):
            if 'edge_label' in data:
                # data['label'] = str(data['edge_label'])  # Permanently visible
                data['title'] = f"Relationship: {data['edge_label']}"  # Hover tooltip

        # Use Pyvis to create an HTML interactive graph
        if html_path:
            if "/" in html_path and not os.path.exists(html_path):
                os.makedirs(os.path.dirname(html_path), exist_ok=True)
            net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
            net.from_nx(G)
            net.save_graph(html_path)

        # FOR LATER: Display in Streamlit
        # with open("interactive_graph.html", 'r', encoding='utf-8') as f:
        #     html_content = f.read()
        # components.html(html_content, height=500)
        
        return G


def generate_color_palette(num_colors):
    """Generate a list of distinct colors."""
    if num_colors <= 10:
        # Use predefined colors for small sets
        predefined_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        return predefined_colors[:num_colors]
    else:
        # Generate colors using HSV color space for larger sets
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            color = mcolors.hsv_to_rgb([hue, 0.7, 0.9])
            hex_color = mcolors.to_hex(color)
            colors.append(hex_color)
        return colors

def test_knowledge_graph(graph_rag, questions=None):
    if not questions:
        questions = [
            "What is Charles Darwin occupation?",
            "What did Alfred Rusel Wallece published?",
            "Where is Charles Darwin buried?"
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

def build_knowledge_graph(llm, database_name, graph_cache):

    # Initialize the GraphRAG system
    print("Initializing GraphRAG system...")
    graph_rag = GraphRAGDefault(llm, database_name=database_name, graph_cache_file=graph_cache)

    
    if graph_rag.graph_cache_file and os.path.exists(graph_rag.graph_cache_file):
        graph_rag.load_graph_docs()
    else:
        documents = get_documents_from_directory(
            directory=Path.home() / "papers_minedd",
            extensions=['.json'],
            chunk_size=20, # Number of sentences to merge into one Document
            overlap=4 # Number of sentences to overlap between chunks
        )
        documents = random.sample(documents, k=50)
        asyncio.run(graph_rag.create_from_documents(documents))
        
    
    # Display schema
    print("\nGraph Schema:")
    print(graph_rag.get_schema())
    
    # Visualize graph in my own HTML (could also use MemGraph console for more powerful exploration)
    graph_rag.visualize_graph(html_path="knowledge_graph.html")

    return graph_rag
    


if __name__ == "__main__":
    from langchain.chat_models import init_chat_model

    model_name="gemini-2.5-flash-lite-preview-06-17"
    model_provider="google_genai"
    ### OR
    # model_name="llama3.2:latest"
    # model_provider="ollama"
    llm = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            temperature=0.0,
            )

    # Example usage for Building a KG
    graph_rag = build_knowledge_graph(llm, database_name="test_papers", graph_cache='graph_gemini_custom_prompt.pkl')


    # Example queries
    # test_knowledge_graph(graph_rag)
