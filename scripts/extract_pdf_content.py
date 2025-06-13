"""

    Execution examples: 
     - Run Grobid only: python scripts/extract_pdf_content.py --directory "/papers_minedd" --mode grobid
     - Run Full pipeline: python scripts/extract_pdf_content.py --directory "/papers_minedd" --llm "llama3.2:latest" --mode all

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
                "use_llm": True,
                "llm_service": "marker.services.ollama.OllamaService",
                "ollama_model": model_llm,  # Specify which model you want to use
                "ollama_base_url": "http://localhost:11434",  # Default Ollama URL,
                "paginate_output": False # Set to True if you need pagination string separators
            }

            if skip_existing and os.path.exists(pdf_json_path):
                print(f"Skipping document since {pdf_json_path} exists already!")
                continue
            elif os.path.exists(pdf_json_path):
                print(f"Loading Document content available at {pdf_json_path}...")
                pdf_paper = DocumentPDF.from_json(pdf_json_path)
                pdf_paper.pdf_path = pdf_path # Rewrite path to current one
            else:
                # Create a DocumentPDF object and load the PDF
                pdf_paper = DocumentPDF(pdf_path, marker_config=marker_config)

            
            # Extract Content as Markdown
            if mode == "all" or mode == "markdown":
                try:
                    pdf_paper.get_markdown()
                except Exception as e:
                    print(f"Could not convert document {pdf_path} to Markdown")
                    print(e)
            
            # Extract Content Chunks as JSON
            if mode == "all" or mode == "grobid":
                try:
                    pdf_paper.get_grobid_chunks(return_as_dict=True, group_dict_by_section=True)
                except Exception as e:
                    print(f"Could not extract {pdf_path} Chunks into JSON content")
                    print(e)

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
    parser.add_argument('--llm', type=str, default="llama3.2:latest", help="The Ollama Model for Marker to use")
    parser.add_argument('--mode', type=str, default="all", help="Which content to extract. Options: ['all', 'grobid', 'markdown', 'tables']")
    parser.add_argument('--skip_existing', action='store_true', help="If true, it will skip the pdfs that already have a json associated")
    args = parser.parse_args()

    directory = args.directory
    model_llm = args.llm
    mode = args.mode

    if mode == 'all':
        skip_existing = True
    else:
        skip_existing=args.skip_existing
    

    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"ERROR: The directory {directory} does not exist.")
        exit()
    
    if args.mode in ['all', 'grobid', 'markdown', 'tables']:
        extract_content_from_pdfs(directory, model_llm, mode, skip_existing)
    else:
        print(f"ERROR: Unknown mode '{args.mode}'. Please choose from ['all', 'grobid', 'markdown', 'tables']")

