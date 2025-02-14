from dotenv import load_dotenv
import argparse
import time
import signal
import sys

from logger_config import configure_logging  # Import the logging configuration
from rag import KnowledgeBase
from web import DirectQueryLangchain  # Import the DirectQuery class
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
    parser = argparse.ArgumentParser(description="Query the local knowledge base.")
    parser.add_argument('--docs', type=str, default='./docs', help='Path to documentation files')
    parser.add_argument('--vector-store', type=str, default='./vector_store', help='Path to vector store')
    parser.add_argument('--history', action='store_true', help='Show conversation history')
    
    args = parser.parse_args()

    history = ConversationHistory()
    kb = KnowledgeBase(args.docs, args.vector_store)
    chat = ChatOpenAI()
    
    thread_id = chat.generate_thread_id()
    chat.set_current_thread_id(thread_id)

     # Register signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    # default to chat mode, use /rag /chat to switch modes
    mode = "chat"
    prompt = "❓>: "

    while True:
        query = input(prompt).strip()
        if query.lower() == 'exit':
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
