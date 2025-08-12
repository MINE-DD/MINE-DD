"""
    To use this script at full capacity, make sure to do the following:
    
    1. Install Ollama and run the Ollama server (to use Marker with LLMs)
    
    2. Install Grobid and run the Grobid server

    Otherwise you can run the script on --simple mode, which will only use marker without LLMs to extract content

    3. Run this script with the desired parameters

    Execution examples: 
     - Run Simplest pipeline (Marker without LLM): python tools/extract_pdf_content.py --directory "/path/to/pdfs/" --mode simple
     - Run Grobid only: python tools/extract_pdf_content.py --directory "/path/to/pdfs/" --mode grobid
     - Run Full pipeline (Grobid + Marker): python tools/extract_pdf_content.py --directory "/path/to/pdfs/" --llm "llama3.2:latest" --mode all

     The converted content will be saved as JSON files in the same directory as the PDFs.

     python tools/extract_pdf_content.py --directory "/Users/jose/papers_campylo" --llm "llama3.2:latest" --mode all --skip_existing

"""

import os
import argparse
from minedd.document import DocumentPDF


def extract_content_from_pdfs(directory: str, model_llm: str, mode:str, skip_existing:bool):
    # Iterate through all files in the directory
    pdf_files = [filename for filename in os.listdir(directory) if filename.endswith('.pdf')]
    n_files = len(pdf_files)
    for i, filename in enumerate(pdf_files):
        if filename.endswith('.pdf'):
            print(f"\n>> Processing {filename} [{i+1}/{n_files}]\n")
            pdf_path = os.path.join(directory, filename)
            pdf_json_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}.json")

            # Define Marker Config (especially LLM to use)
            marker_config = {
                "output_format": "markdown",
                "use_llm": False,
                "paginate_output": False # Set to True if you need pagination string separators
            }
            if model_llm is not None and mode != "simple":
                marker_config["use_llm"] = True
                marker_config["llm_service"] = "marker.services.ollama.OllamaService"
                marker_config["ollama_base_url"] = "http://localhost:11434"  # Default Ollama URL
                marker_config["ollama_model"] = model_llm

            if skip_existing and os.path.exists(pdf_json_path):
                print(f"Skipping document since {pdf_json_path} exists already!")
                continue
            elif os.path.exists(pdf_json_path):
                print(f"Loading Document content available at {pdf_json_path}...")
                pdf_paper = DocumentPDF.from_json(pdf_json_path)
                if pdf_paper is not None:
                    pdf_paper.pdf_path = pdf_path # Rewrite path to current one
            else:
                # Create a DocumentPDF object and load the PDF
                pdf_paper = DocumentPDF(pdf_path, marker_config=marker_config)

            if pdf_paper is None:
                print(f"ERROR: Could not load PDF {pdf_path}. Skipping...")
                continue

            # Extract Content as Markdown
            if mode == "simple" or mode == "all" or mode == "markdown":
                try:
                    pdf_paper.get_markdown()
                except Exception as e:
                    print(f"Could not convert document {pdf_path} to Markdown")
                    print(e)
            
            # Extract MAIN Content Chunks as JSON
            if mode == "all" or mode == "grobid":
                try:
                    pdf_paper.get_grobid_chunks(return_as_dict=True, group_dict_by_section=True)
                except Exception as e:
                    print(f"Could not extract {pdf_path} Chunks into JSON content")
                    print(e)
            elif mode == "simple":
                text_chunks = pdf_paper.get_chunks(mode="chars")
                doc_title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
                docs_dict = {"title": doc_title, "grouped_by_section": False,"chunks": []}
                for i, chunk in enumerate(text_chunks):
                    docs_dict["chunks"].append({"text": chunk, "pages": None, "chunk_index": i, "section":"NoTitle", "section_number": -1})
                pdf_paper.json_content = docs_dict

            # Extract Tables
            if mode == "all" or mode == "tables":
                try:
                    pdf_paper.get_document_tables()
                except Exception as e:
                    print(f"Could not extract Tables from {pdf_path}")
                    print(e)

            # Save the extracted text to a text file
            try:
                pdf_paper.to_json(pdf_json_path)
            except Exception as e:
                print(f"Could not save DocumentPDF to {pdf_json_path}")
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text content from PDFs in a directory.")
    parser.add_argument('--directory', type=str, help="The path to the directory containing PDF files.")
    parser.add_argument('--llm', type=str, default=None, help="The Ollama Model for Marker to use")
    parser.add_argument('--mode', type=str, default="simple", help="Which content to extract. Options: ['all', 'grobid', 'markdown', 'tables']")
    parser.add_argument('--skip_existing', action='store_true', help="If true, it will skip the pdfs that already have a json associated")
    args = parser.parse_args()

    directory = args.directory
    model_llm = args.llm
    mode = args.mode

    skip_existing=args.skip_existing

    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"ERROR: The directory {directory} does not exist.")
        exit()
    
    if args.mode in ['simple', 'all', 'grobid', 'markdown', 'tables']:
        extract_content_from_pdfs(directory, model_llm, mode, skip_existing)
    else:
        print(f"ERROR: Unknown mode '{args.mode}'. Please choose from ['simple', 'all', 'grobid', 'markdown', 'tables']")

