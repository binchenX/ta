import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma  # Updated import
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings

from logger_config import configure_logging  # Import the logging configuration

logger = configure_logging()

class DirectQuery:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=os.getenv('MODEL_NAME', 'gpt-4o-mini_v2024-07-18'),
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )

    def query(self, query_text: str) -> str:
        logger.debug(f"Querying directly with LLM: {query_text}")
        chain = LLMChain(llm=self.llm)
        response = chain.invoke({"query": query_text})
        logger.debug(f"Response: {response['result']}")
        return response["result"]
