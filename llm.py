import os
from typing import Optional

from openai import OpenAI


class LLM:
    """
    A class that provides language model capabilities for text processing tasks.

    This class wraps OpenAI's API to provide high-level functions for text operations
    such as proofreading and summarization.

    Attributes:
        client (OpenAI): An instance of the OpenAI client
        model (str): The identifier of the OpenAI model to use
    """

    def __init__(self, model: str, client: Optional[OpenAI] = None, api_key: Optional[str] = None):
        """
        Initialize the LLM class.

        Args:
            model (str): The identifier of the model to use (e.g., "gpt-4")
            client (Optional[OpenAI]): An authenticated OpenAI client instance. If not provided,
                                     will create a new client using api_key or environment variable.
            api_key (Optional[str]): OpenAI API key. If not provided, will look for OPENAI_API_KEY
                                   environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment variables
        """
        if client:
            self.client = client
        else:
            # Try to get API key from parameter or environment variable
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No OpenAI API key provided. Either pass an API key, "
                    "provide a client instance, or set the OPENAI_API_KEY environment variable."
                )
            self.client = OpenAI(api_key=api_key)

        self.model = model

    def proofread(self, text: str) -> str:
        """
        Proofreads the given text for spelling, grammar, and clarity improvements.

        This method uses the OpenAI model to analyze the input text and provide
        specific suggestions for improvements. It includes both the original text
        and a list of recommended changes in its response.

        Args:
            text (str): The text to be proofread

        Returns:
            str: A string containing the proofreading results, including:
                - The original text
                - Specific suggestions for improvements
                - Any identified errors and their corrections
                If an error occurs, returns an error message string

        Examples:
            >>> llm = LLM("gpt-4", api_key="your-api-key")
            >>> result = llm.proofread("This is a exemple text with misspellings.")
            >>> print(result)
            Original text: "This is a exemple text with misspellings."
            Suggestions:
            1. Replace "exemple" with "example"
            ...

        Raises:
            Exception: Handled internally, returns error message as string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a proofreading assistant. Analyze the given text and provide specific suggestions for spelling, grammar, and clarity improvements. Include the original text and list changes in your response.",
                    },
                    {"role": "user", "content": f"Proofread this: {text}"},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"

    def summarize(self, text: str, max_length: int = 150) -> str:
        """
        Summarizes the given text using the OpenAI model.

        Args:
            text (str): The text to summarize
            max_length (int): Maximum desired length of the summary in words (default: 150)

        Returns:
            str: The summarized text or error message

        Examples:
            >>> llm = LLM("gpt-4", api_key="your-api-key")
            >>> summary = llm.summarize("Long article text...", max_length=50)
            >>> print(summary)
            "Concise summary of the article..."
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a summarizing assistant. Create a clear and concise summary of the following text in no more than {max_length} words. Maintain the key points while eliminating unnecessary details.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"


if __name__ == "__main__":
    # Load environment variables if using python-dotenv
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize LLM
    llm = LLM("gpt-4o-mini_v2024-07-18")

    # Example 1: Proofreading
    text_to_proofread = """
    Their are several misteakes in this sentense. Its not writen very good, 
    and it definately needs to be checked for spelling and grammer errors.
    """
    print("\nProofreading example:")
    print("Original text:", text_to_proofread)
    print("\nResult:", llm.proofread(text_to_proofread))

    # Example 2: Summarization
    text_to_summarize = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals.
    """
    print("\nSummarization example:")
    print("Original text:", text_to_summarize)
    print("\nResult:", llm.summarize(text_to_summarize, max_length=30))
