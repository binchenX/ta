import os
import sys

from openai import OpenAI

from intent import IntentInferrer
from log import configure_logging

logger = configure_logging()


class ProofReadAgent:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini_v2024-07-18"
        self.inferrer = IntentInferrer(api_key=api_key, model=self.model)
        if not self.client.api_key:
            raise ValueError("No OpenAI API key provided or found in env.")

    def _proofread_with_openai(self, text, detailed=False):
        try:
            if detailed:
                system_content = (
                    "You are a proofreader. Fix spelling, grammar, and clarity. "
                    "Return in Markdown: '## Original text\n> text\n"
                    "## Proofreaded text\n> text\n## Changes\n changes'."
                )
            else:
                system_content = (
                    "You are a proofreader. Fix spelling, grammar, and clarity. "
                    "Return proofread text in '> text' format."
                )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"Proofread this: {text}"},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"

    def _read_file(self, file_path):
        try:
            if not os.path.exists(file_path):
                return f"Error: File '{file_path}' not found."
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def proofread_file(self, file_path, detailed=False):
        file_content = self._read_file(file_path)
        if "Error" in file_content:
            return file_content
        return self._proofread_with_openai(file_content, detailed)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python proofread_agent.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    agent = ProofReadAgent()
    # Simple mode
    response = agent.proofread_file(filename, detailed=False)
    print("Simple mode:")
    print(response)
    # Detailed mode
    response = agent.proofread_file(filename, detailed=True)
    print("\nDetailed mode:")
    print(response)
