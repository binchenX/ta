from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma  # Updated import
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings

import argparse

from logger_config import configure_logging  # Import the logging configuration
from rag import KnowledgeBase
from history import ConversationHistory

# Load environment variables from .env file
load_dotenv()
logger = configure_logging()


def main():
    parser = argparse.ArgumentParser(description="Query the local knowledge base.")
    parser.add_argument('--query', type=str, help='The query to ask the knowledge base.')
    parser.add_argument('--docs', type=str, default='./docs', help='Path to documentation files')
    parser.add_argument('--vector-store', type=str, default='./vector_store', help='Path to vector store')
    parser.add_argument('--history', action='store_true', help='Show conversation history')
    
    args = parser.parse_args()

    # Initialize conversation history
    history = ConversationHistory()

    if args.history:
        print("\nConversation History:")
        for timestamp, query, response in history.get_history():
            print(f"\nTime: {timestamp}")
            print(f"Q: {query}")
            print(f"A: {response}")
        return

    if not args.query:
        parser.print_help()
        return

    # Initialize knowledge base
    kb = KnowledgeBase(args.docs, args.vector_store)
    
    # Query knowledge base
    try:
        response = kb.query(args.query)
        print(f"\nQ: {args.query}")
        print(f"A: {response}")
        
        # Save conversation
        history.save(args.query, response)
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
