import os
import logging
from typing import List
from typing import Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings

from logger_config import configure_logging

# Disable PostHog analytics
os.environ["POSTHOG_API_KEY"] = ""

# Disable ChromaDB telemetry
chroma_settings = Settings(anonymized_telemetry=False, allow_reset=True)

logger = configure_logging()


class KnowledgeBase:
    def __init__(
        self, doc_paths: List[str], vector_store_path: str, rag_config: Dict[str, str]
    ):
        logging.info("Initializing KnowledgeBase")

        # Initialize OpenAI API
        self.llm_api_key = os.getenv("OPENAI_API_KEY")
        if not self.llm_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Load configuration values, giving precedence to environment variables
        self.llm_base_url = os.getenv("OPENAI_BASE_URL", rag_config.get("llm_base_url"))
        self.model_name = rag_config.get("model_name")
        self.embedding_name = rag_config.get("embedding_name")
        # todo - use one doc only for now
        self.docs_path = doc_paths[0]

        # log all rg configration values
        logger.info(f"llm_base_url: {self.llm_base_url}")
        logger.info(f"model_name: {self.model_name}")
        logger.info(f"embedding_name: {self.embedding_name}")
        logger.info(f"docs_path: {self.docs_path}")

        # Initialize language model
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_name,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )

        # Initialize embeddings with the specified model
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_name,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )

        self.vector_store_path = vector_store_path

        # Create directories if they don't exist
        os.makedirs(self.docs_path, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)

        self.vector_store = self._initialize_vector_store()

        # Initialize retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.vector_store.as_retriever()
        )
        logging.info("KnowledgeBase initialized successfully")

    def _initialize_vector_store(self, reindex: bool = False):
        logging.debug("Initializing vector store")

        # Check if reindex is True or if vector store exists and has documents
        if (
            not reindex
            and os.path.exists(self.vector_store_path)
            and len(os.listdir(self.vector_store_path)) > 0
        ):
            logging.info("Loading existing vector store")
            return Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                client_settings=chroma_settings,
            )

        # Reindex is True or vector store does not exist or is empty

        # Clear existing collection before indexing
        if (
            Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                client_settings=chroma_settings,
            )
            .get()
            .get("ids")
        ):
            Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                client_settings=chroma_settings,
            ).delete_collection()

        # Load and process documents
        loader = DirectoryLoader(self.docs_path, glob="**/*.md")
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Create and persist vector store
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path,
            client_settings=chroma_settings,
        )
        # vector_store.persist()
        logging.info("Vector store initialized and persisted")
        return vector_store

    def reindex(self):
        logging.info("Reindexing knowledge base")
        self.vector_store = self._initialize_vector_store(reindex=True)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.vector_store.as_retriever()
        )
        logging.info("Knowledge base reindexed successfully")

    def query(self, query_text: str) -> str:
        logging.debug(f"Querying knowledge base with: {query_text}")
        response = self.qa_chain.invoke({"query": query_text})
        logging.debug(f"Response: {response['result']}")
        return response["result"]
