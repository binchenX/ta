from dotenv import load_dotenv
import argparse
import time

from logger_config import configure_logging  # Import the logging configuration
from rag import KnowledgeBase
from web import DirectQueryLangchain  # Import the DirectQuery class
from history import ConversationHistory
from chat import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
logger = configure_logging()

def main():
    parser = argparse.ArgumentParser(description="Query the local knowledge base.")
    parser.add_argument('--docs', type=str, default='./docs', help='Path to documentation files')
    parser.add_argument('--vector-store', type=str, default='./vector_store', help='Path to vector store')
    parser.add_argument('--history', action='store_true', help='Show conversation history')
    
    args = parser.parse_args()

    history = ConversationHistory()
    #kb = KnowledgeBase(args.docs, args.vector_store)
    #dq = DirectQueryLangchain()
    chat = ChatOpenAI()
    
    thread_id = chat.generate_thread_id()
    chat.set_current_thread_id(thread_id)

    while True:
        query = input("❓>: ").strip()
        if query.lower() == 'exit':
            print("Exiting interactive query mode. Goodbye!")
            break

        if not query:
            print("Please enter a valid question.")
            continue

        # Determine which query function to use
        # if query.startswith('/rag'):
        #     query = query[len('/rag'):].strip()
        #     query_function = kb.query
        # else:
        #     query_function = chat.query

        # Query knowledge base or direct query
        try:
            response = chat.query(query)
            print(f"🤖: {response}\n")
            
            # Save conversation
            history.save(query, response)
        
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
