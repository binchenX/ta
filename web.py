import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from logger_config import configure_logging  # Import the logging configuration

logger = configure_logging()

import os

class DirectQueryLangchain:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=os.getenv('MODEL_NAME', 'gpt-4o-mini_v2024-07-18'),
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        
    def query(self, query_text: str) -> str:
        logger.debug(f"Querying directly with LLM: {query_text}")

        # Define the prompt template
        prompt_template = PromptTemplate(
            template="Answer the following question: {query}",
            input_variables=["query"]
        )

        # Construct the sequence
        sequence = prompt_template | self.llm

        # Execute the sequence
        response = sequence.invoke({"query": query_text})
        logger.debug(f"Response: {response}")

         # Check if response is an instance of AIMessage and retrieve the content
        if hasattr(response, 'content'):
            return response.content
        else:
            logger.error("Invalid response format received")
            return "Error: Invalid response format received"
