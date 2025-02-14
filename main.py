import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # Updated import
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sqlite3
import argparse

class KnowledgeBase:
    def __init__(self, docs_path: str, vector_store_path: str):
        # Initialize OpenAI API
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.openai_base_url = os.getenv('OPENAI_BASE_URL')
        
        # Initialize language model
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini_v2024-07-18",
            api_key=self.openai_api_key,
            base_url=self.openai_base_url
        )
        
        # Initialize embeddings with the specified model
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-3-large_v1',  # Updated model name
            api_key=self.openai_api_key,
            base_url=self.openai_base_url
        )
        
        # Initialize or load vector store
        self.vector_store_path = vector_store_path
        self.docs_path = docs_path
        
        # Create directories if they don't exist
        os.makedirs(self.docs_path, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        self.vector_store = self._initialize_vector_store()
        
        # Initialize retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

    def _initialize_vector_store(self):
        # Check if vector store exists and has documents
        if os.path.exists(self.vector_store_path) and len(os.listdir(self.vector_store_path)) > 0:
            return Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
        
        # Check if there are any markdown files
        markdown_files = [f for f in os.listdir(self.docs_path) if f.endswith('.md')]
        if not markdown_files:
            raise FileNotFoundError(f"No markdown files found in {self.docs_path}")
        
        # Load and process documents
        loader = DirectoryLoader(self.docs_path, glob="**/*.md")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create and persist vector store
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )
        vector_store.persist()
        return vector_store

    def query(self, query_text: str) -> str:
        """Query the knowledge base"""
        # Updated to use invoke instead of run
        response = self.qa_chain.invoke({"query": query_text})
        return response["result"]  # Updated to get result from response

# ... rest of the code remains the same ...

class ConversationHistory:
    def __init__(self, db_path: str = 'conversation_history.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS history 
            (timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, query TEXT, response TEXT)
        ''')
        self.conn.commit()

    def save(self, query: str, response: str):
        self.cursor.execute(
            "INSERT INTO history (query, response) VALUES (?, ?)",
            (query, response)
        )
        self.conn.commit()

    def get_history(self, limit: int = 10):
        self.cursor.execute(
            "SELECT timestamp, query, response FROM history ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        return self.cursor.fetchall()

    def __del__(self):
        self.conn.close()

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
