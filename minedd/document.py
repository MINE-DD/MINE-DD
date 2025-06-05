import re
import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from langchain_text_splitters.character import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter


class DocumentPDF:
    def __init__(self, pdf_path: str, marker_config:dict=None, output_format:str="markdown"):
        # Basic Attributes
        self.pdf_path = pdf_path
        self.markdown = None
        self.json_content = None
        # Configure Marker to use Ollama as the LLM service
        if marker_config is None:
            self.marker_config = {
                "output_format": output_format,  # Default output format is markdown, can also be 'json'?
                "use_llm": True,
                "llm_service": "marker.services.ollama.OllamaService",
                "ollama_model": "llama3.2:latest",  # Specify which model you want to use
                "ollama_base_url": "http://localhost:11434",  # Default Ollama URL,
                "paginate_output": False # Set to True if you need pagination string separators
            }
        else:
            # Ensure the model is set correctly in the config
            self.marker_config = marker_config

        # Create config parser
        config_parser = ConfigParser(self.marker_config)
        self.marker_converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )

    def get_markdown(self) -> str:
        if self.markdown is not None:
            return self.markdown
        else:
            if self.marker_config.get("output_format") != "markdown":
                raise ValueError("Output format must be set to 'markdown' in the configuration.")
            try:
                # Convert PDF to markdown
                rendered = self.marker_converter(self.pdf_path)
                # Extract the markdown text and images
                marker_text, _, images = text_from_rendered(rendered)
                self.markdown = marker_text
                return marker_text
            except Exception as e:
                raise RuntimeError(f"Failed to read PDF file: {e}")
    
    def get_json(self) -> str:
        if self.json_content is not None:
            return self.json_content
        else:
            if self.marker_config.get("output_format") != "json":
                raise ValueError("Output format must be set to 'json' in the configuration.")
            try:
                # Convert PDF to JSON
                rendered = self.marker_converter(self.pdf_path)
                return rendered
            except Exception as e:
                raise RuntimeError(f"Failed to read PDF file: {e}")


class DocumentMarkdown:
    def __init__(self, md_content:str = None, md_path: str = None):
        # Basic Attributes
        self.name = None
        self.md_path = md_path if md_path else None
        if md_content and md_path:
            print("WARNING! Both 'md_content' and 'md_path' were provided, only loading the content from 'md_content'.")
        if md_content:
            self.markdown = md_content
        elif md_path:
            self.markdown = self.load_content(self.md_path)
        else:
            raise ValueError("Either 'md_content' or 'md_path' must be provided.")     
    
    def get_title(self) -> str:
        """
        Extracts the title from the markdown content.
        The title is assumed to be the first header (level 1) in the markdown.
        """
        if self.markdown is None:
            return ""
        # Match the first header (level 1)
        match = re.search(r'^#\s*(.*)', self.markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return ""

    def get_markdown(self, only_text:bool=False, remove_markup:bool=False, remove_references:bool=False) -> str:
        if self.markdown is None:
            return ""
        
        text = self.markdown
        if remove_references:
            text = text.partition("# References")[0]
        if remove_markup:
            text = re.sub(r'^#+\s.*\n?', '', text) # Remove headers (lines starting with #)
            text = re.sub(r'<sup>.+</sup>', '', text)  # Remove footnote indicators
            text = re.sub(r'<br>', ' ', text)  # Remove emphasis
        if only_text:
            # Remove markdown formatting to return plain text
            text = re.sub(r'\*\*(.*?)\*\*|\*(.*?)\*|`(.*?)`|!\[(.*?)\]\((.*?)\)|\[(.*?)\]\((.*?)\)', r'\1\2\3\4', text)
            text = text.replace('\n', ' ').strip()
        return text

    def load_content(self, from_path: str):
        try:
            with open(self.md_path, 'r', encoding='utf-8') as file:
                self.markdown = file.read()
                return self.markdown
        except FileNotFoundError:
            print(f"WARNING: Markdown file not found at '{from_path}'")
            return None
    
    def save_content(self, to_path: str):
        if self.markdown is not None:
            # Create path if not exists
            if not os.path.exists(to_path):
                os.makedirs(os.path.dirname(to_path), exist_ok=True)
            # Save the content
            with open(to_path, 'w', encoding='utf-8') as file:
                file.write(self.markdown)
        else:
            print("WARNING: No markdown content to save.")
    
    def convert_to_chunks(self, mode:str='chars', chunk_size:int=1000, overlap:int=100):
        valid_opts = ['chars', 'newlines', 'sentences']
        if mode not in valid_opts:
            raise ValueError(f"Mode must be one of {valid_opts}, got '{mode}' instead.")
        if mode == 'sentences':
            text = self.get_markdown(remove_references=True, remove_markup=True)
            splitter = SentenceTransformersTokenTextSplitter()
        elif mode == 'chars':
            text = self.get_markdown(only_text=True, remove_markup=True, remove_references=True)
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", " ", ""])
        elif mode == 'newlines':
            text = self.get_markdown(remove_references=True, remove_markup=True)
            splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separator="\n\n", length_function=len)
        
        chunks = splitter.split_text(text)
        return chunks
    
    def get_paper_reference(self):
        """
        Extracts the paper reference from the markdown content.
        """
        if self.markdown is None:
            return None
        # Extract the first line as the paper reference
        ref_text = self.markdown.partition("# References")[1]
        ref_list = ref_text.split('\n')
        return ref_list

