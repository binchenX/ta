from dotenv import load_dotenv
import argparse
import signal
import sys
import os
import readline

from logger_config import configure_logging
from rag import KnowledgeBase
from history import ConversationHistory
from chat import ChatOpenAI

from rich.console import Console
from rich.markdown import Markdown

# Initialize the rich console
console = Console(width=100)

# Load environment variables from .env file
load_dotenv()
logger = configure_logging()


def signal_handler(sig, frame):
    print("\nExiting interactive query mode. Goodbye!")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="iTelligent Assistant (ta)")
    args = parser.parse_args()

    # configrations
    base_path = os.path.expanduser("~/.ta")
    vector_store_path = os.path.join(base_path, "vector_store")
    os.makedirs(vector_store_path, exist_ok=True)
    history_db_path = os.path.join(base_path, "history.db")
    threads_path = os.path.join(base_path, "threads.json")
    # for thread similarity matching (experimental)
    chroma_db_path = os.path.join(base_path, "chroma_db")

    rag_enabled = False
    rag_docs_path = os.getenv("RAG_DOC_PATH")

    # if RAG_DOC_PATH is not set, don't enable RAG mode
    if rag_docs_path:
        rag_enabled = True
        kb = KnowledgeBase(docs_path=rag_docs_path, vector_store_path=vector_store_path)

    chat = ChatOpenAI(save_file=threads_path, chroma_db_path=chroma_db_path)
    history = ConversationHistory(db_path=history_db_path)

    thread_id = chat.generate_thread_id()
    chat.set_current_thread_id(thread_id)

    # Register signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    # default to chat mode, use /rag /chat to switch modes
    mode = "chat"

    while True:
        if mode == "rag":
            prompt = "[rag] ❓>: "
        else:
            prompt = "❓>: "

        query = input(prompt).strip()

        # if query is /help, show help
        if query.lower() == "/help":
            print("Commands:")
            print("/lt: List threads")
            print("/st [thread_id]: Set current thread")
            print("/dt [thread_id]: Delete thread")
            print("/nt: New thread")
            print("/rag: Switch to RAG mode")
            print("/chat: Switch to chat mode")
            print("/rag reindex: Reindex knowledge base")
            print("/exit: Exit interactive query mode")
            continue

        if query.lower() == "exit":
            print("Exiting interactive query mode. Goodbye!")
            break

        # list threads
        if query.lower() == "/lt":
            threads = chat.list_threads_with_topics()
            print("Threads:")
            for thread in threads:
                print(f"- {thread}")
            continue

        # set curent thread
        if query.lower().startswith("/st"):
            try:
                thread_id = query.split()[1]
                chat.set_current_thread_id(thread_id)
                print(f"Current thread set to {thread_id}")
            except (IndexError, ValueError):
                print("Invalid thread ID. Please enter a valid number after 'st'.")
            continue

        # delete thread by id
        if query.lower().startswith("/dt"):
            try:
                thread_id = query.split()[1]
                if chat.delete_thread(thread_id):
                    print(f"Deleted thread {thread_id}")
                else:
                    print(f"Thread {thread_id} not found")
            except (IndexError, ValueError):
                print("Invalid thread ID. Please enter a valid number after 'dt'.")
            continue

        # new thread
        if query.lower() == "/nt":
            thread_id = chat.generate_thread_id()
            chat.set_current_thread_id(thread_id)
            print(f"New thread created with ID: {thread_id}")
            continue

        # switch to rag mode if query starts with /rag
        if query.lower().strip() == "/rag":
            if not rag_enabled:
                print("RAG mode is not enabled. Please set RAG_DOC_PATH in .env file.")
                continue
            else:
                print("Switching to RAG mode...")
                mode = "rag"
            continue

        # if query is /rag reindex, reindex the knowledge base
        if query.lower() == "/rag reindex":
            kb.reindex()
            continue

        if query.lower().strip() == "/chat":
            print("Switching to chat mode...")
            mode = "chat"
            continue

        if not query:
            print("Please enter a valid question.")
            continue

        # Query knowledge base or direct query
        try:
            if mode == "rag":
                response = kb.query(query)
            elif mode == "chat":
                response = chat.query(query)

            print(f"🤖:\n")

            markdown_response = Markdown(response)
            console.print(markdown_response)
            # Save conversation
            history.save(query, response)

        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
