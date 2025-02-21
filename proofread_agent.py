import os
import json
import sys
from openai import OpenAI
from log import configure_logging

logger = configure_logging()


class ProofReadAgent:
    def __init__(self, api_key=None):
        """Initialize the handcrafted proofreading agent with OpenAI client."""
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini_v2024-07-18"
        if not self.client.api_key:
            raise ValueError("OpenAI API key not provided or found in environment.")

    def _infer_intent_and_file(self, user_input):
        """Use OpenAI to infer proofreading intent and extract file name."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": 'You are an intent detector. Determine if the user wants to proofread a file and extract the file name from the input. Respond with valid JSON using double quotes, e.g., {"intent": "proofread", "file": "filename"} if proofreading is intended, or {"intent": "none"} if not.',
                    },
                    {"role": "user", "content": user_input},
                ],
                max_tokens=50,
                temperature=0.5,
            )
            raw_content = response.choices[0].message.content.strip()
            logger.info(f"Intent inference response (raw): {raw_content}")
            if "'" in raw_content and '"' not in raw_content:
                raw_content = raw_content.replace("'", '"')
            intent_data = json.loads(raw_content)
            logger.info(f"Parsed intent_data: {intent_data}")
            return intent_data
        except Exception as e:
            logger.error(f"Error inferring intent: {str(e)}")
            return {"intent": "none", "error": f"Error inferring intent: {str(e)}"}

    def _proofread_with_openai(self, text):
        """Send text to OpenAI for proofreading with detailed suggestions."""
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

    def _read_file(self, file_path):
        """Manually read a file from the local filesystem."""
        try:
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' not found."
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def proofread(self, user_input):
        """Process user input, infer intent, and proofread the file if applicable."""
        intent_data = self._infer_intent_and_file(user_input)
        logger.info(f"Evaluating intent_data in proofread: {intent_data}")
        if intent_data.get("intent") == "proofread" and "file" in intent_data:
            file_path = intent_data["file"]
            logger.info(f"Proofreading file: {file_path}")
            file_content = self._read_file(file_path)
            if "Error" in file_content:
                return file_content
            return self._proofread_with_openai(file_content)
        return "Sorry, I couldn’t understand your request. Try asking to proofread a file, e.g., 'Check example.txt'."


# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python proofread_agent.py <filename>")
        sys.exit(1)

    # Construct a natural language input from the command line argument
    filename = sys.argv[1]
    user_input = f"Please proofread {filename}"

    agent = ProofReadAgent()
    response = agent.proofread(user_input)
    print(response)
