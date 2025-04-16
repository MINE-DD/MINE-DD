"""
Command-line interface for the MINEDD package.

CLI commands for interacting with the package via a terminal
"""


import argparse
import sys
import os
import warnings
import logging

from minedd.query import Query
from minedd.embeddings import Embeddings

# suppressing warnings and messages from paper-qa and pydantic
warnings.filterwarnings("ignore")
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("paperqa").setLevel(logging.ERROR)

def add_embed_args(parser_embed):
    
    parser_embed.add_argument(
        '--llm',
        type=str,
        default='ollama/llama3.2:1b',
        help='LLM model to use (default is llama3.2:1b)'
    )

    parser_embed.add_argument(
        '--embedding_model',
        type=str,
        default='ollama/mxbai-embed-large:latest',
        help='Embedding model to use (default is mxbai-embed-large:latest)'
    )

    parser_embed.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Directory to save outputs (default: out)"
    )

    parser_embed.add_argument(
        "--paper_directory",
        type=str,
        default="data/",
        help="Directory containing paper files (default: data/)"
    )

    parser_embed.add_argument(
        '--embeddings_filename',
        type=str,
        default='embeddings.pkl',
        help='Filename for the embeddings pickle file (default: embeddings.pkl)'
    )

    parser_embed.add_argument(
        '--augment_existing',
        action='store_true',
        help='Augment existing embeddings if they exist'
    )

    return parser_embed

def add_query_args(parser_query):
    parser_query.add_argument(
        '--llm',
        type=str,
        default='ollama/llama3.2:1b',
        help='LLM model to use (default is llama3.2:1b)'
    )

    parser_query.add_argument(
        '--embedding_model',
        type=str,
        default='ollama/mxbai-embed-large:latest',
        help='Embedding model to use (default is mxbai-embed-large:latest)'
    )

    parser_query.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Directory to save outputs (default: out)"
    )

    parser_query.add_argument(
        "--paper_directory",
        type=str,
        default="data/",
        help="Directory containing paper files (default: data/)"
    )

    parser_query.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to the embeddings pickle file'
    )

    parser_query.add_argument(
        "--questions_file",
        type=str,
        help="Path to Excel file with questions"
    )

    parser_query.add_argument(
        "--question",
        type=str,
        help="Single question to ask"
    )

    parser_query.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum number of retries for model loading (default: 2)"
    )

    return parser_query

def embed_command(args):
    """ Handle the embed command"""

    emb_path = os.path.join(args.output_dir, args.embeddings_filename)

    # Create the Embeddings object
    engine = Embeddings(
        model=args.llm,
        embedding_model=args.embedding_model,
        paper_directory=args.paper_directory,
        output_embeddings_path=emb_path,
        existing_docs=None
    )

    # Load existing embeddings if desired and provided
    if args.augment_existing:
        if not os.path.exists(emb_path):
            print(f"Warning: {emb_path} does not exist. Creating from scratch.")
        else:
            engine.load_existing_embeddings(emb_path)
            print(f"Found valid {emb_path} file. Augmenting it...")
    elif os.path.exists(emb_path):
            print(f"Warning: {emb_path} already exists. Overwriting it...")

    # Process the papers and create embeddings
    pdf_file_list = engine.prepare_papers()
    print(f"Found {len(pdf_file_list)} PDF files in {args.paper_directory}.")
    if len(pdf_file_list) == 0:
        print(f"No PDF files found in {args.paper_directory}.")
        sys.exit(1)
    else:
        engine.process_papers(pdf_file_list)

def query_command(args):
    """ Handle the query command"""
    engine = Query(
        model=args.llm,
        embedding_model=args.embedding_model,
        paper_directory=args.paper_directory,
        output_dir=args.output_dir
    )

    engine.load_embeddings(args.embeddings)

    # either process a single question or a batch of questions
    if args.questions_file:
        questions = engine.load_questions(args.questions_file)
        output_file = f"{args.output_dir}/questions_with_answers_{args.llm.split('/')[-1].replace(':','_').replace('/','_')}.xlsx"
        engine.query_batch(
            questions,
            save_individual=True,
            output_file=output_file,
            max_retries=args.max_retries
        )
    elif args.question: # this is useful for debugging for example, or for real-time interactions
        try:
            result = engine.query_single(args.question, max_retries=args.max_retries)
            print("\n=== Question ===")
            print(result['question'])
            print("\n=== Answer ===")
            print(result['answer'])
            print("\n=== Sources ===")
            for i, citation in enumerate(result['citations']):
                print(f"{i+1}. {citation}")
            if result['urls'] and len(result['urls']) > 0:
                print("\n=== URLs ===")
                for i, url in enumerate(result['urls']):
                    print(f"{i+1}. {url}")
        except Exception as e:
            print(f"Error processing query: {e}")
            sys.exit(1)
    else: # in case something is wrong
        print("Error: Either --questions_file or --question must be provided.")
        sys.exit(1)


def main():
    # Create Main Parser
    parser = argparse.ArgumentParser(description='MINEDD: Query scientific papers with LLMs')

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title='subcommands', 
        description='Available subcommands: embed | query', 
        dest='command',
        required=True
    )
    
    # Add arguments specific for the Embed command
    parser_embed = subparsers.add_parser(
        "embed", help="Create the paper embeddings in your filesystem"
        )
    parser_embed.set_defaults(func=embed_command)
    parser_embed = add_embed_args(parser_embed)
    
    # Add arguments specific for the Query command
    parser_query = subparsers.add_parser(
        "query", help="Query for existing paper embeddings to obtain answers to questions"
        )
    parser_query.set_defaults(func=query_command)
    parser_query = add_query_args(parser_query)
  
    # Parse arguments and call the corresponding command
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
