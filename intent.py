import json
import os
import unittest

from openai import OpenAI

from log import configure_logging

logger = configure_logging()


class IntentInferrer:
    def __init__(self, api_key=None, model="gpt-4o-mini_v2024-07-18"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        if not self.client.api_key:
            raise ValueError("No OpenAI API key provided or found in env.")

    def infer_intent_and_file(self, user_input):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You detect user intent. Supported intents: 'proofread' (with file name, "
                            "check for 'detailed' or 'simple', default simple), 'fetchnews', or 'none'. "
                            "Return JSON, e.g., {'intent': 'proofread', 'file': 'file.txt', 'detailed': true}, "
                            "{'intent': 'fetchnews'}, or {'intent': 'none'}. Extendable for future intents."
                        ),
                    },
                    {"role": "user", "content": user_input},
                ],
                max_tokens=50,
                temperature=0.5,
            )
            raw_content = response.choices[0].message.content.strip()
            logger.info(f"Intent inference raw response: {raw_content}")
            if "'" in raw_content and '"' not in raw_content:
                raw_content = raw_content.replace("'", '"')
            intent_data = json.loads(raw_content)
            logger.info(f"Parsed intent_data: {intent_data}")
            return intent_data
        except Exception as e:
            logger.error(f"Error inferring intent: {str(e)}")
            return {"intent": "none", "error": str(e)}


class TestIntentInferrer(unittest.TestCase):
    def setUp(self):
        self.inferrer = IntentInferrer()

    def test_proofread_simple(self):
        result = self.inferrer.infer_intent_and_file("Proofread file.txt")
        expected = {"intent": "proofread", "file": "file.txt", "detailed": False}
        self.assertEqual(result, expected)

    def test_proofread_detailed(self):
        result = self.inferrer.infer_intent_and_file("Proofread file.txt with detailed review")
        expected = {"intent": "proofread", "file": "file.txt", "detailed": True}
        self.assertEqual(result, expected)

    def test_proofread_simple_explicit(self):
        result = self.inferrer.infer_intent_and_file("Proofread file.txt simply")
        expected = {"intent": "proofread", "file": "file.txt", "detailed": False}
        self.assertEqual(result, expected)

    def test_fetchnews(self):
        result = self.inferrer.infer_intent_and_file("Fetch some news")
        expected = {"intent": "fetchnews"}
        self.assertEqual(result, expected)

    def test_none_intent(self):
        result = self.inferrer.infer_intent_and_file("Hello world")
        expected = {"intent": "none"}
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
