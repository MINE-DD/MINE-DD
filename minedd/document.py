import re
import os
import json
import pandas as pd
# For Grobid-Based PDF Text extraction
from langchain_community.document_loaders.parsers import GrobidParser
from langchain_community.document_loaders.generic import GenericLoader
# For Chunking Texts
from langchain_core.documents import Document
from langchain_text_splitters.character import CharacterTextSplitter, RecursiveCharacterTextSplitter


class DocumentPDF:
    def __init__(self, pdf_path: str, marker_config:dict=None, output_format:str="markdown"):
        # Basic Attributes
        self.pdf_path = pdf_path
        self.markdown = None
        self.json_content = None
        self.tables = []
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
            self.marker_config = marker_config
        self.marker_converter = None
        

    def _init_marker(self):
        """Initialize Marker only when needed"""

        if self.marker_converter is None:
            # Import at runtime For Marker as PDF to Markdown Converter
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
            from marker.config.parser import ConfigParser
            try:
                config_parser = ConfigParser(self.marker_config)
                self.marker_converter = PdfConverter(
                    config=config_parser.generate_config_dict(),
                    artifact_dict=create_model_dict(),
                    processor_list=config_parser.get_processors(),
                    renderer=config_parser.get_renderer(),
                    llm_service=config_parser.get_llm_service()
                )
                self.text_from_rendered = text_from_rendered
                print("Marker initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Marker: {e}")
                self.marker_converter = None

    @classmethod
    def from_json(cls, json_path: str):
        try:
            with open(json_path) as f:
                content = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {json_path}. Returning None")
            return None
        except json.decoder.JSONDecodeError:
            print(f"File {json_path} is empty or not a valid JSON! Returning None")
            return None          

        try:
            doc = cls(json_path.replace('.json', '.pdf'))  # Assuming the PDF file has the same name as the JSON file
            doc.markdown = content["markdown"]
            doc.json_content = content["text_chunks"]
            doc.tables = [pd.DataFrame(t) for t in content["tables_as_json"]]
            return doc
        except Exception as e:
            print(f"Problem loading {json_path}. Error {e}.\nCheck that the file has the right format. Returning None")
            return None


    def get_markdown(self) -> str:
        if self.markdown is not None:
            return self.markdown
        else:
            if self.marker_config.get("output_format") != "markdown":
                raise ValueError("Output format must be set to 'markdown' in the configuration.")
            try:
                self._init_marker()  # Initialize only when actually needed
                # Convert PDF to markdown
                rendered = self.marker_converter(self.pdf_path)
                # Extract the markdown text and images
                marker_text, _, images = self.text_from_rendered(rendered)
                self.markdown = marker_text
                return marker_text
            except Exception as e:
                raise RuntimeError(f"Failed to read PDF file: {e}")
    
    def get_grobid_chunks(self, 
                            segment_sentences:bool = True, 
                            return_as_dict:bool = False, 
                            group_dict_by_section:bool = True) -> dict | list[Document]:
        """
        uses GROBID (instructions here: https://grobid.readthedocs.io/en/latest/Install-Grobid/) 
        to extract chunks from the PDF document and returns and organized JSON per paper section
        """
        try:
            loader = GenericLoader.from_filesystem(
                self.pdf_path,
                parser= GrobidParser(segment_sentences=segment_sentences)
            )
            docs = loader.load()
        except Exception as e:
            print("ERROR", e)
            print("Returning empty list...")
            docs = []
        
        if len(docs) == 0:
            return docs

        if return_as_dict and group_dict_by_section:
            docs_dict = {
                "title": docs[0].metadata.get("paper_title", "NoTitle"),
                "grouped_by_section": group_dict_by_section,
                "sections_titles": [],
                "sections_content": {}
                }
            for i, doc in enumerate(docs):
                pages_str = "-".join((re.findall(r"'([^']+)'", doc.metadata['pages'])))
                section_title = doc.metadata.get("section_title", "NoSection")
                chunk_index = i if segment_sentences else int(doc.metadata['para'])
                row = {
                    "text": doc.page_content.replace('\n', ' '),
                    "pages": pages_str,
                    "chunk_index": chunk_index,
                    "section":section_title,
                    "section_number": doc.metadata.get("section_number", -1)
                }
                if section_title not in docs_dict["sections_titles"]:
                    docs_dict["sections_titles"].append(section_title)
                    docs_dict["sections_content"][section_title] = []
                
                docs_dict["sections_content"][section_title].append(row)
            docs = docs_dict
            self.json_content = docs_dict
        elif return_as_dict:
            docs_dict = {
                "title": docs[0].metadata.get("paper_title", "NoTitle"),
                "grouped_by_section": group_dict_by_section,
                "chunks": [],
                }
            for i, doc in enumerate(docs):
                pages_str = "-".join((re.findall(r"'([^']+)'", doc.metadata['pages'])))
                section_title = doc.metadata.get("section_title", "NoSection")
                chunk_index = i if segment_sentences else int(doc.metadata['para'])
                row = {
                    "text": doc.page_content.replace('\n', ' '),
                    "pages": pages_str,
                    "chunk_index": chunk_index,
                    "section":section_title,
                    "section_number": doc.metadata.get("section_number", -1)
                }
                docs_dict["chunks"].append(row)
            docs = docs_dict
            self.json_content = docs_dict
        return docs


    def get_document_tables(self) -> list[str]:
        "Uses GMT to extract tables as loadable CSV files from the PDF document. Stores the tables in a list of loadable strings."
        # For Table Extraction
        from gmft.auto import AutoTableDetector, AutoTableFormatter
        from gmft.pdf_bindings import PyPDFium2Document
        
        detector = AutoTableDetector()
        formatter = AutoTableFormatter()
        doc = PyPDFium2Document(self.pdf_path)
        tables = []
        for page in doc:
            tables += detector.extract(page)
        formatted_tables = [formatter.extract(table) for table in tables]
        # Store them as a List of Pandas DataFrames
        dataframes = []
        for index, table in enumerate(formatted_tables):
            dataframes.append(table.df())
        self.tables = dataframes
        return dataframes


    def get_chunks(self, mode:str='chars', chunk_size:int=1000, overlap:int=100, as_langchain_docs:bool=False) -> list[str] | list[Document]:
        """_summary_

        Args:
            mode (str, optional): Type of Chunking to be done. Defaults to 'chars'.
            chunk_size (int, optional): Chunk size normally in characters but if mode is 'from_json' then units are sentences. Defaults to 1000.
            overlap (int, optional): Chunk overlap normally in characters but if mode is 'from_json' then units are sentences. Defaults to 100.
            as_langchain_docs (bool, optional): Wether to return already langchain Document objects. Defaults to False.

        Raises:
            ValueError: Checks if the mode is valid. If not, raises an error.

        Returns:
            list[str] | list[Document]: List of text chunks or LangChain Document objects depending on the `as_langchain_docs` flag.
        """
        if mode not in ['chars', 'from_json']:
            raise ValueError("Chunk extraction NOT supported! Mode must be 'chars' (from Markdown) or 'from_json' (From Grobid Generated JSON).")
        if mode == 'from_json':
            filename = os.path.basename(self.pdf_path)
            docs = process_from_grobid_chunks(filename, self.json_content, as_langchain_docs=as_langchain_docs, chunk_size=chunk_size, overlap=overlap)
            return docs
        else:
            md_doc = DocumentMarkdown(md_content=self.get_markdown())
            chunks = md_doc.convert_to_chunks(mode='chars', chunk_size=chunk_size, overlap=overlap)
            if as_langchain_docs:
                docs = []
                for i, chunk in enumerate(chunks):
                    docs.append(Document(
                        page_content= chunk,
                        metadata={"chunk_index": i, "filepath": self.pdf_path}
                    ))
                return docs
            else:
                return chunks
    
    def get_document_as_dict(self):
        return {
            "original_path": self.pdf_path,
            "markdown": self.markdown,
            "text_chunks": self.json_content,
            "tables_as_json": [t.to_dict() for t in self.tables]
        }
    
    def to_json(self, json_path: str):
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.get_document_as_dict(), json_file, indent=2)


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
        valid_opts = ['chars', 'newlines']
        if mode not in valid_opts:
            raise ValueError(f"Mode must be one of {valid_opts}, got '{mode}' instead.")
        elif mode == 'chars':
            text = self.get_markdown(only_text=True, remove_markup=True, remove_references=True)
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True, separators=["\n\n", "\n", ".", "!", "?", " ", ""])
        elif mode == 'newlines':
            text = self.get_markdown(remove_references=True, remove_markup=True)
            splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separator="\n\n", length_function=len)
        
        chunks = splitter.split_text(text)
        return chunks
    
    def get_paper_references(self):
        """
        Extracts the paper references from the markdown content.
        """
        if self.markdown is None:
            return None
        # Extract the first line as the paper reference
        ref_text = self.markdown.partition("# References")[1]
        ref_list = ref_text.split('\n')
        return ref_list




def process_from_grobid_chunks(filename, json_content, as_langchain_docs:bool=False, chunk_size:int=1000, overlap:int=100):

    def _process_json_batch(batch):
        if as_langchain_docs:
            text = " ".join([c['text'] for c in batch])
            metadata = {
                "source": filename,
                "title": doc_title,
                "pages": f"{batch[0]['pages'].split('-')[0]}-{batch[-1]['pages'].split('-')[-1]}",
                "chunk_index": f"{batch[0]['chunk_index']}-{batch[-1]['chunk_index']}",
                "section": batch[0]['section'],
                "sentence_count": len(batch),
            }
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            return doc
        else:
            return [c['text'] for c in batch]
    # json_content = self.json_content
    if json_content is None:
        raise ValueError("No JSON content available. Please run 'get_grobid_chunks' first.")
    print("Processing JSON content into chunks of size ", chunk_size, " sentences with overlap ", overlap)
    doc_title = json_content['title']
    is_grouped_by_section = json_content['grouped_by_section']
    chunks_key = 'sections_content' if is_grouped_by_section else 'chunks'
    step_size = max(1, chunk_size - overlap)  # Ensure step_size >= 1
    documents = []
    if is_grouped_by_section:
        for section_name, sentences in json_content[chunks_key].items():
            for i in range(0, len(sentences), step_size):
                batch = sentences[i:i + chunk_size]
                doc = _process_json_batch(batch)
                documents.append(doc)
    else:
        for i in range(0, len(json_content[chunks_key]), step_size):
            batch = json_content[chunks_key][i:i + chunk_size]
            doc = _process_json_batch(batch)
            documents.append(doc)
    return documents


def get_documents_from_directory(directory, extensions=['.json'], chunk_size=10, overlap=5) -> list[Document]:
    """Pre-process a directory of documents and load them into the vector store"""
    # Load documents from directory
    documents = []
    processed_files = 0
    for filename in os.listdir(directory):
        if '.json' in extensions and filename.endswith('.json'):
            processed_files += 1
            full_paper = DocumentPDF.from_json(json_path=os.path.join(directory, filename))
            paper_chunks = full_paper.get_chunks(
                mode='from_json', 
                chunk_size=chunk_size, 
                overlap=overlap, 
                as_langchain_docs=True
                )
            documents.extend(paper_chunks)

    print(f"Processed {processed_files} files, extracted {len(documents)} Document chunks")
    return documents