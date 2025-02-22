import argparse
import os
import signal
import sys
from typing import Callable, Dict

from dotenv import load_dotenv

from cmd_chat import run_interactive_chat
from config import Config
from log import configure_logging
from proofread_agent import ProofReadAgent
from rag import KnowledgeBase

# Load environment variables from .env file
load_dotenv()
logger = configure_logging()


def signal_handler(sig, frame):
    print("\nExiting interactive query mode. Goodbye!")
    sys.exit(0)


def run_proofread(filename: str) -> str:
    logger.debug(f"Proofreading file: {filename}")
    agent = ProofReadAgent()
    return agent.proofread_file(filename)


def handle_proofread(args):
    print(run_proofread(args.filename))
    sys.exit(0)


def handle_model_list(args, config):
    print("Models:")
    for model in config.models:
        alias = config.get_alias_from_model(model)
        print(f"- {model} ({alias})")
    sys.exit(0)


def handle_model_set(args, config):
    if config.set_chat_model(args.model_name):
        print(f"Successfully changed chat model to: {config.get_chat_model()}")
    else:
        print(f"Failed to change model - '{args.model_name}' is not a valid model")
    sys.exit(0)


def handle_model_help(args, parser):
    parser.print_help()
    sys.exit(1)


def handle_rag_index(args, kb):
    kb.reindex(kb.doc_paths, force_reindex=args.force)
    print("Indexing completed." if not args.force else "Forced reindexing completed.")
    sys.exit(0)


def handle_rag_addpath(args, config, kb, config_file):
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist.")
        sys.exit(1)
    config.rag["rag_doc_paths"].append(args.path)
    config.save_config(config_file)
    kb.reindex([args.path], force_reindex=False)
    print(f"Added and indexed path: {args.path}")
    sys.exit(0)


def handle_rag_help(args, parser):
    parser.print_help()
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="iTelligent Assistant (ta)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: proofread (with alias 'pf')
    proofread_parser = subparsers.add_parser("proofread", aliases=["pf"], help="Proofread a file")
    proofread_parser.add_argument("filename", help="Path to the file to proofread")

    # Subcommand: model
    model_parser = subparsers.add_parser("model", help="Manage chat models")
    model_subparsers = model_parser.add_subparsers(dest="model_command", help="Model subcommands")

    # Subcommand: model list
    model_subparsers.add_parser("list", help="List available models and aliases")

    # Subcommand: model set
    set_parser = model_subparsers.add_parser("set", help="Set the chat model")
    set_parser.add_argument("model_name", help="Model name to set as current chat model")

    # Subcommand: rag
    rag_parser = subparsers.add_parser("rag", help="Manage RAG indexing")
    rag_subparsers = rag_parser.add_subparsers(dest="rag_command", help="RAG subcommands")

    # Subcommand: rag index
    index_parser = rag_subparsers.add_parser("index", help="Index directories to knowledge base")
    index_parser.add_argument(
        "-f", "--force", action="store_true", help="Force reindexing of all directories"
    )

    # Subcommand: rag addpath
    addpath_parser = rag_subparsers.add_parser("addpath", help="Add a new path to RAG and index it")
    addpath_parser.add_argument("path", help="Path to be added and indexed")

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    base_path = os.path.expanduser("~/.ta")
    vector_store_path = os.path.join(base_path, "vector_store")
    os.makedirs(vector_store_path, exist_ok=True)
    config_file = (
        "config.toml" if os.path.exists("config.toml") else os.path.join(base_path, "config.toml")
    )
    config = Config(config_file)

    # Command dispatch dictionary with alias mapping
    command_handlers: Dict[str, Callable] = {
        "proofread": lambda: handle_proofread(args),
        "pf": lambda: handle_proofread(args),  # Explicitly map alias
        "model": {
            "list": lambda: handle_model_list(args, config),
            "set": lambda: handle_model_set(args, config),
            None: lambda: handle_model_help(args, model_parser),
        },
        "rag": {
            "index": lambda: handle_rag_index(args, kb),
            "addpath": lambda: handle_rag_addpath(args, config, kb, config_file),
            None: lambda: handle_rag_help(args, rag_parser),
        },
    }

    # Initialize dependencies only if needed
    kb = (
        KnowledgeBase(
            doc_paths=config.rag["rag_doc_paths"],
            vector_store_path=vector_store_path,
            rag_config=config.rag,
        )
        if args.command == "rag" and os.path.exists(vector_store_path)
        else None
    )

    # Execute the appropriate command handler
    handler = command_handlers.get(args.command)
    if handler:
        if isinstance(handler, dict):
            subcommand = getattr(args, f"{args.command}_command", None)
            handler = handler.get(subcommand, handler[None])
        handler()
    else:
        run_interactive_chat()


if __name__ == "__main__":
    main()
