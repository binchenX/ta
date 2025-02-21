# math_agent.py
import openai
import json
import os
from log import configure_logging

logger = configure_logging()


class MathAgent:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini_v2024-07-18"
        logger.info(f"Initialized with model: {self.model}")

    def get_functions(self):
        return [
            {
                "name": "add",
                "description": "Add two numbers together and add 100 to the result. The result should be used as input for subsequent operations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "First number"},
                        "y": {"type": "number", "description": "Second number"},
                    },
                    "required": ["x", "y"],
                },
            },
            {
                "name": "multiply",
                "description": "Multiply a number by 2. Can be used with the result from the add function.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "Number to multiply, can be result from previous operation",
                        },
                    },
                    "required": ["x"],
                },
            },
        ]

    def add(self, x: float, y: float) -> dict:
        result = x + y + 100
        logger.info(f"Add function: {x} + {y} + 100 = {result}")
        return {"result": result}

    def multiply(self, x: float) -> dict:
        result = x * 2
        logger.info(f"Multiply function: {x} * 2 = {result}")
        return {"result": result}

    def chat(self, user_message: str) -> str:
        logger.info(f"\nNew chat with input: {user_message}")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful math assistant. When using functions, use the result from previous function calls.",
            },
            {"role": "user", "content": user_message},
        ]

        while True:
            logger.info("Making API call")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[{"type": "function", "function": f} for f in self.get_functions()],
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message
            logger.info(f"Assistant message: {assistant_message}")

            # Break if no more tool calls
            if not hasattr(assistant_message, "tool_calls") or not assistant_message.tool_calls:
                return assistant_message.content

            messages.append(assistant_message)

            # Process tool calls
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                logger.info(f"Calling {function_name} with args: {function_args}")

                function_response = getattr(self, function_name)(**function_args)
                logger.info(f"Function response: {function_response}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_response),
                    }
                )


if __name__ == "__main__":
    chat = MathAgent()
    test_messages = [
        "Add 5 and 3, then multiply the result by 2",
        "What's 10 plus 20 times 2?",
        "First multiply 5 by 2, then add it to 10",
    ]

    for message in test_messages:
        response = chat.chat(message)
        logger.info(f"Final response: {response}")
