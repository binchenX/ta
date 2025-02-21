import os
from mcp.client.sse import sse_client
from mcp import ClientSession
from openai import OpenAI
import asyncio
import json
from log import configure_logging
from typing import List, Optional
from mcp import Tool  # Assuming Tool is imported from mcp

logger = configure_logging()


async def select_tool_by_description(
    tools: List[Tool],
    client: OpenAI,
    model: str,
    purpose: str,
    max_tokens: int = 50,
    temperature: float = 0.5,
) -> Optional[Tool]:
    """Select a tool from a list based on its description using an LLM."""
    if not tools:
        logger.info("No tools provided for selection.")
        return None

    # Prepare tool descriptions for LLM
    tools_descriptions = [
        f"Tool '{tool.name}': {tool.description or 'No description provided'}" for tool in tools
    ]
    tools_text = "\n".join(tools_descriptions)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a tool selector. Given a list of tools with descriptions, identify which tool is designed to {purpose}. "
                        'Return the tool name in JSON format, e.g., {"tool_name": "tool_name_here"}. '
                        'If no suitable tool is found, return {"tool_name": null}.'
                    ),
                },
                {"role": "user", "content": tools_text},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw_content = response.choices[0].message.content.strip()
        logger.info(f"LLM tool selection response: {raw_content}")
        result = json.loads(raw_content)
        selected_tool_name = result.get("tool_name")

        # Find the matching tool
        for tool in tools:
            if tool.name == selected_tool_name:
                return tool
        return None
    except Exception as e:
        logger.error(f"Error selecting tool with LLM: {str(e)}")
        return None


class ProofReadAgentMCP:
    def __init__(self, api_key=None, mcp_uri="http://127.0.0.1:8000/sse"):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided or found in environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini_v2024-07-18"
        self.mcp_uri = mcp_uri
        # Discover tools during initialization
        self.file_read_tool = asyncio.run(self._discover_tools())
        if not self.file_read_tool:
            logger.warning("No file-reading tool found during initialization.")
        else:
            logger.info(
                f"Initialized with file-reading tool: {self.file_read_tool.name} - {self.file_read_tool.description}"
            )

    async def _discover_tools(self):
        """Discover available MCP tools and select a file-reading tool."""
        async with sse_client(url=self.mcp_uri) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                logger.info(f"Available tools: {tools_result}")
                return await select_tool_by_description(
                    tools=tools_result.tools,
                    client=self.client,
                    model=self.model,
                    purpose="read file contents",
                )

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
            if not self.file_read_tool:
                return "No tool available to read files."
            async with sse_client(url=self.mcp_uri) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    response = await session.call_tool(self.file_read_tool.name, {"file": file})
                    if response.isError:
                        logger.error(f"Error calling tool: {response.content[0].text}")
                        return f"Error: {response.content[0].text}"
                    else:
                        logger.info(f"Response: {response}")
                        file_content = response.content[0].text
                        proofread_result = self._proofread_with_openai(file_content)
                        return proofread_result
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
