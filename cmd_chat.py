import atexit
import os
import readline
import sys

from rich.console import Console
from rich.markdown import Markdown

from chat import ChatOpenAI
from config import Config
from hackernews import HackerNews
from history import ConversationHistory
from intent import IntentInferrer
from log import configure_logging
from proofread_agent import ProofReadAgent
from rag import KnowledgeBase

# Configure readline
HISTFILE = os.path.expanduser("~/.ta_history")
try:
    readline.read_history_file(HISTFILE)
    readline.set_history_length(1000)
except FileNotFoundError:
    pass

# Save history on exit
atexit.register(readline.write_history_file, HISTFILE)

# Enable tab completion
readline.parse_and_bind("tab: complete")

logger = configure_logging()
# Initialize the rich console
console = Console(width=120)


def get_input(prompt):
    """Helper function to get input with proper line editing capabilities"""
    try:
        return input(prompt).strip()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return ""
    except EOFError:
        print("\nEOF detected. Exiting...")
        sys.exit(0)


def run_interactive_chat():
    base_path = os.path.expanduser("~/.ta")
    vector_store_path = os.path.join(base_path, "vector_store")
    os.makedirs(vector_store_path, exist_ok=True)
    history_db_path = os.path.join(base_path, "history.db")
    threads_path = os.path.join(base_path, "threads.json")
    chroma_db_path = os.path.join(base_path, "chroma_db")

    config_file = "config.toml"
    if not os.path.exists(config_file):
        config_file = os.path.join(base_path, "config.toml")

    config = Config(config_file)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    if not openai_api_key or not openai_base_url:
        print("Error: OPENAI_API_KEY and OPENAI_BASE_URL must be set in the environment.")
        sys.exit(1)

    rag_enabled = False
    rag_doc_paths = config.rag["rag_doc_paths"]
    if rag_doc_paths:
        valid_paths = [path for path in rag_doc_paths if os.path.exists(path)]
        if valid_paths:
            rag_enabled = True
            logger.info(f"RAG enabled with doc paths: {valid_paths}")
            kb = KnowledgeBase(
                doc_paths=valid_paths,
                vector_store_path=vector_store_path,
                rag_config=config.rag,
            )

    chat = ChatOpenAI(
        save_file=threads_path,
        chroma_db_path=chroma_db_path,
        chat_config=config.chat,
    )

    history = ConversationHistory(db_path=history_db_path)
    thread_id = chat.generate_thread_id()
    chat.set_current_thread_id(thread_id)
    mode = "chat"

    inferrer = IntentInferrer(api_key=openai_api_key, model=config.get_chat_model())

    while True:
        model_alias = config.get_alias_from_model(
            config.get_chat_model() if mode == "chat" else config.get_rag_model()
        )
        prompt = f"[rag] ({model_alias}) ❓>: " if mode == "rag" else f"({model_alias}) ❓>: "
        query = get_input(prompt)

        if query.lower() == "/help":
            print("Commands:")
            print("/lm: List models")
            print("/lt: List threads")
            print("/st [thread_id]: Set current thread")
            print("/dt [thread_id]: Delete thread")
            print("/nt: New thread")
            print("/chat: Switch to chat mode")
            print("/rag: Switch to RAG mode")
            print("/exit: Exit interactive query mode")
            continue

        if query.lower() == "exit":
            print("Exiting interactive query mode. Goodbye!")
            break

        if query.lower() == "/lm":
            print("Models:")
            for model in config.models:
                alias = config.get_alias_from_model(model)
                print(f"- {model} ({alias})")
            continue

        if query.lower() == "/lt":
            threads = chat.list_threads_with_topics()
            print("Threads:")
            for thread in threads:
                print(f"- {thread}")
            continue

        if query.lower().startswith("/st"):
            try:
                thread_id = query.split()[1]
                chat.set_current_thread_id(thread_id)
                print(f"Current thread set to {thread_id}")
            except (IndexError, ValueError):
                print("Invalid thread ID. Please enter a valid number after 'st'.")
            continue

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

        if query.lower() == "/nt":
            thread_id = chat.generate_thread_id()
            chat.set_current_thread_id(thread_id)
            print(f"New thread created with ID: {thread_id}")
            continue

        if query.lower().strip() == "/rag":
            if not rag_enabled:
                print("RAG mode is not enabled. Please set RAG_DOC_PATH in .env file.")
                continue
            else:
                print("Switching to RAG mode...")
                mode = "rag"
            continue

        if query.lower().strip() == "/chat":
            print("Switching to chat mode...")
            mode = "chat"
            continue

        if not query:
            print("Please enter a valid question.")
            continue

        try:
            # infer intent first
            intent_data = inferrer.infer_intent_and_file(query)
            if intent_data.get("intent") == "proofread" and "file" in intent_data:
                file_path = intent_data["file"]
                response = ProofReadAgent().proofread_file(file_path, intent_data["detailed"])
                print("🤖:\n")
                console.print(Markdown(response))
            elif intent_data.get("intent") == "fetchnews":
                print("Fetching news...")
                print("🤖:\n")
                HackerNews().do()
            else:
                if mode == "rag":
                    response = kb.query(query)
                elif mode == "chat":
                    response = chat.query(query)
                print("🤖:\n")
                markdown_response = Markdown(response)
                console.print(markdown_response)
                history.save(query, response)
        except Exception as e:
            print(f"Error: {str(e)}")
