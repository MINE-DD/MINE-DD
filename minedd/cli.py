"""
Command-line interface for the MINEDD package.

CLI commands for interacting with the package via a terminal
"""


import argparse
import sys
from minedd.query import Query
def query_command(args):
    """ Handle the query command"""
    engine = Query(
        model=args.llm,
        embedding_model=args.embedding_model,
        output_dir=args.output_dir
    )

    engine.load_embeddings(args.embeddings)

    # either process a single question or a batch of questions
    if args.questions_file:
        engine.load_questions(args.questions_file)
        output_file = f"{args.output_dir}/questions_with_answers_{args.llm.split('/')[-1].replace(':','_').replace('/','_')}.xlsx"
        engine.query_batch(questions, save_individual=True, output_file=output_file)
    elif args.question: # this is useful for debugging for example, or for real-time interactions
        result = engine.query_single(args.question)
        print("\n=== Question ===")
        print(result['question'])
        print("\n=== Answer ===")
        print(result['answer'])
        print("\n=== Sources ===")
        for i, citation in enumerate(result['citations']):
            print(f"{i+1}. {citation}")
    else: # in case something is wrong
        print("Error: Either --questions_file or --question must be provided.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='MINEDD: Mining scientific papers with LLMs')
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # commands
    query_parser = subparsers.add_parser('query', help='Query papers with questions')
    query_parser.add_argument(
        '--llm',
        type=str,
        default='ollama/llama3.2:1b',
        help='LLM model to use (default is llama3.2:1b)'
    )

    query_parser.add_argument(
        '--embedding_model',
        type=str,
        default='ollama/mxbai-embed-large:latest',
        help='Embedding model to use (default is mxbai-embed-large:latest)'
    )

    query_parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to the embeddings pickle file'
    )

    query_parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Directory to save outputs (default: out)"
    )

    query_parser.add_argument(
        "--questions_file",
        type=str,
        help="Path to Excel file with questions"
    )
    query_parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask"
    )

    # parse commands and call functions

    args = parser.parse_args()

    if args.command == "query":
        query_command(args)
    else: # in case commands are not right
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
