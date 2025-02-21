import os
from mcp.client.sse import sse_client
from mcp import ClientSession
from openai import OpenAI
import asyncio
import json
from log import configure_logging

logger = configure_logging()


def find_tool_by_name(tools, name):
    for tool in tools:
        if tool.name == name:
            return tool
    return None


class ProofReadAgentMCP:
    def __init__(self, api_key=None, mcp_uri="http://127.0.0.1:8000/sse"):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided or found in environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini_v2024-07-18"
        self.mcp_uri = mcp_uri

    def _infer_intent_and_file(self, user_input):
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
            if "'" in raw_content and '"' not in raw_content:
                raw_content = raw_content.replace("'", '"')
            intent_data = json.loads(raw_content)
            return intent_data
        except Exception as e:
            print(f"Error inferring intent: {str(e)}")
            return {"intent": "none", "error": f"Error inferring intent: {str(e)}"}

    def _proofread_with_openai(self, text):
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

    async def proofread_async(self, user_input):
        intent_data = self._infer_intent_and_file(user_input)
        if intent_data.get("intent") == "proofread" and "file" in intent_data:
            file = intent_data["file"]
            async with sse_client(url=self.mcp_uri) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    logger.info(f"Available tools: {tools_result}")
                    read_file_tool = find_tool_by_name(tools_result.tools, "read_file")
                    if read_file_tool:
                        # todo: deal with error
                        response = await session.call_tool("read_file", {"file": file})
                        if response.isError:
                            logger.error(f"error calling tools {response.content[0].text}")
                        else:
                            logger.info(f"response {response}")
                            file_content = response.content[0].text
                            proofread_result = self._proofread_with_openai(file_content)
                            return proofread_result
                    else:
                        return "Required tools are not available."
        return "Sorry, I couldn't understand your request. Try asking to proofread a file, e.g., 'Check example.txt'."

    def proofread(self, user_input):
        return asyncio.run(self.proofread_async(user_input))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python proofread_agent_mcp.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    user_input = f"Please proofread {filename}"
    agent = ProofReadAgentMCP()
    response = agent.proofread(user_input)
    print(response)
