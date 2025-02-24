import json
import os

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
                            "You detect user intent. Supported intents: 'proofread' (with file name), "
                            "'fetchnews', or 'none'. Return JSON, e.g., "
                            '{"intent": "proofread", "file": "file.txt"} or {"intent": "fetchnews"} '
                            'or {"intent": "none"}. Extendable for future intents.'
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
