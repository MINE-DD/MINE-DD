"""
Command-line interface for the MINEDD package.

CLI commands for interacting with the package via a terminal
"""


import argparse
import sys
import warnings
import logging

from minedd.query import Query

# suppressing warnings and messages from paper-qa and pydantic
warnings.filterwarnings("ignore")
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("paperqa").setLevel(logging.ERROR)

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
    parser = argparse.ArgumentParser(description='MINEDD: Query scientific papers with LLMs')
    
    # Add query arguments directly to the main parser
    parser.add_argument(
        '--llm',
        type=str,
        default='ollama/llama3.2:1b',
        help='LLM model to use (default is llama3.2:1b)'
    )

    parser.add_argument(
        '--embedding_model',
        type=str,
        default='ollama/mxbai-embed-large:latest',
        help='Embedding model to use (default is mxbai-embed-large:latest)'
    )

    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to the embeddings pickle file'
    )

    parser.add_argument(
        "--paper_directory",
        type=str,
        default="data/",
        help="Directory containing paper files (default: data/)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Directory to save outputs (default: out)"
    )

    parser.add_argument(
        "--questions_file",
        type=str,
        help="Path to Excel file with questions"
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask"
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum number of retries for model loading (default: 2)"
    )

    # Parse arguments and call the query command directly
    args = parser.parse_args()
    query_command(args)

if __name__ == "__main__":
    main()
