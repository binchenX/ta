import os
import json
from typing import List, Dict
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
    INDEX_RECORD_FILE = "index_record.json"

    def __init__(
        self, doc_paths: List[str], vector_store_path: str, rag_config: Dict[str, str]
    ):
        logger.info("Initializing KnowledgeBase")
        self.llm_api_key = os.getenv("OPENAI_API_KEY")
        if not self.llm_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.llm_base_url = os.getenv("OPENAI_BASE_URL", rag_config.get("llm_base_url"))
        self.model_name = rag_config.get("model_name")
        self.embedding_name = rag_config.get("embedding_name")
        self.doc_paths = doc_paths
        self.vector_store_path = vector_store_path

        logger.info(f"llm_base_url: {self.llm_base_url}")
        logger.info(f"model_name: {self.model_name}")
        logger.info(f"embedding_name: {self.embedding_name}")
        logger.info(f"doc_paths: {self.doc_paths}")

        self.llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_name,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_name,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.index_record_file = os.path.join(
            self.vector_store_path, self.INDEX_RECORD_FILE
        )
        self.indexed_dirs = self.load_index_record()
        self.vector_store = self._initialize_vector_store()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.vector_store.as_retriever()
        )
        logger.info("KnowledgeBase initialized successfully")

    def load_index_record(self):
        if os.path.exists(self.index_record_file):
            with open(self.index_record_file, "r") as f:
                return json.load(f)
        return []

    def save_index_record(self):
        with open(self.index_record_file, "w") as f:
            json.dump(self.indexed_dirs, f)

    def _initialize_vector_store(self, reindex: bool = False):
        logger.debug("Initializing vector store")
        vector_store = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embeddings,
            client_settings=chroma_settings,
        )

        if (
            reindex
            or not os.path.exists(self.vector_store_path)
            or len(os.listdir(self.vector_store_path)) == 0
        ):
            logger.info("Reindexing or initializing new vector store")
            # Clear the existing collection in case of reindexing
            existing_ids = vector_store.get().get("ids", [])
            if existing_ids:
                vector_store.delete(ids=existing_ids)

            # Remove all documents from the vector store if present
            for doc_path in self.doc_paths:
                self.index_directory(doc_path, vector_store)
            self.save_index_record()
        return vector_store

    def index_directory(self, doc_path, vector_store):
        if doc_path in self.indexed_dirs:
            logger.info(f"Directory already indexed: {doc_path}")
            return

        logger.info(f"Indexing directory: {doc_path}")
        loader = DirectoryLoader(doc_path, glob="**/*.md")

        # List to store successfully loaded documents
        valid_documents = []
        # List to store paths of documents that cause errors
        error_documents = []

        try:
            documents = loader.load()
        except Exception as e:
            logger.error(
                f"Failed to load documents from directory: {doc_path}, error: {e}"
            )
            return

        logger.info(f"Loaded {len(documents)} documents from {doc_path}")

        for doc in documents:
            try:
                # Manually split the document to ensure we catch errors per document
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                splits = text_splitter.split_documents([doc])
                valid_documents.extend(splits)
            except Exception as e:
                error_documents.append(doc.metadata["source"])
                logger.error(
                    f"Failed to process document {doc.metadata['source']}, error: {e}"
                )

        if valid_documents:
            vector_store.add_documents(documents=valid_documents)
            self.indexed_dirs.append(doc_path)
            logger.info(f"Indexed and added directory to index record: {doc_path}")
        else:
            logger.warning(f"No valid documents found in directory: {doc_path}")

        if error_documents:
            logger.warning(f"Documents with errors: {error_documents}")

    def reindex(self, doc_paths: List[str], force=False):
        """Reindex the knowledge base using new doc paths, if force is True, index from scratch; otherwise, add new directories"""
        logger.debug(
            f"Reindexing knowledge base with new doc paths: {doc_paths}, force: {force}"
        )
        if force:
            self.doc_paths = doc_paths
            self.vector_store = self._initialize_vector_store(reindex=True)
        else:
            for doc_path in doc_paths:
                if doc_path not in self.indexed_dirs:
                    self.index_directory(doc_path, self.vector_store)
            self.save_index_record()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.vector_store.as_retriever()
        )
        logger.info("Knowledge base reindexed successfully")

    def query(self, query_text: str) -> str:
        logger.debug(f"Querying knowledge base with: {query_text}")
        response = self.qa_chain.invoke({"query": query_text})
        logger.debug(f"Response: {response['result']}")
        return response["result"]
