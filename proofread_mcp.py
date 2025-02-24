import asyncio
import json
import os
from typing import Optional

from openai import OpenAI

from intent import IntentInferrer
from log import configure_logging
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

logger = configure_logging()


class ToolConverter:
    @staticmethod
    def to_openai_functions(mcp_tools):
        """
        Convert MCP tool definitions to OpenAI-compatible function definitions.
        Args:
            mcp_tools (list): A list of MCP tool objects. Each tool object must have 'name', 'description', and 'input_schema' attributes.
        Returns:
            list: A list of OpenAI-compatible function definitions.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.input_schema["properties"],
                    "required": tool.input_schema.get("required", []),
                },
            }
            for tool in mcp_tools
        ]

    @staticmethod
    def to_anthropic_functions(mcp_tools):
        """
        Convert MCP tool definitions to Anthropic-compatible function definitions.
        Args:
            mcp_tools (list): A list of MCP tool objects. Each tool object must have 'name', 'description', and 'input_schema' attributes.
        Returns:
            list: A list of Anthropic-compatible function definitions.
        """
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in mcp_tools
        ]


class MPCToolSelect:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        script_path: Optional[str] = None,
        mcp_uri: Optional[str] = None,
    ):
        self.client = client
        self.model = model
        self.script_path = script_path or os.getenv("MCP_SCRIPT_PATH", "filesystem.py")
        self.mcp_uri = mcp_uri or os.getenv("MCP_URI", "http://127.0.0.1:8000/sse")
        self.use_stdio = bool(script_path)  # Prefer Stdio if script_path is provided

    async def discover_and_select_tool(
        self, purpose: str, max_tokens: int = 50, temperature: float = 0.5
    ) -> Optional[Tool]:
        """Discover available MCP tools and select one based on purpose using an LLM."""
        # Discover tools based on transport type
        if self.use_stdio:
            server_params = StdioServerParameters(
                command="python", args=[self.script_path], env=None
            )
            async with stdio_client(server_params) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    logger.info(f"Stdio tools: {tools_result}")
                    tools = tools_result.tools
        else:
            async with sse_client(url=self.mcp_uri) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    logger.info(f"SSE tools: {tools_result}")
                    tools = tools_result.tools

        # Select tool from discovered list
        if not tools:
            logger.info("No tools provided for selection.")
            return None

        tools_descriptions = [
            f"Tool '{tool.name}': {tool.description or 'No description provided'}" for tool in tools
        ]
        tools_text = "\n".join(tools_descriptions)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
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

            for tool in tools:
                if tool.name == selected_tool_name:
                    return tool
            return None
        except Exception as e:
            logger.error(f"Error selecting tool with LLM: {str(e)}")
            return None


class ProofReadLLM:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def proofread(self, text: str) -> str:
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


class ProofReadAgentMCP:
    def __init__(self, api_key=None, script_path=None, mcp_uri=None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided or found in environment.")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini_v2024-07-18"
        self.tool_selector = MPCToolSelect(self.client, self.model, script_path, mcp_uri)
        self.use_stdio = self.tool_selector.use_stdio
        self.proofreader = ProofReadLLM(self.client, self.model)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided or found in environment.")
        self.inferrer = IntentInferrer(api_key=self.api_key, model=self.model)

        self.file_read_tool = asyncio.run(
            self.tool_selector.discover_and_select_tool("read file contents")
        )
        if not self.file_read_tool:
            logger.warning("No file-reading tool found during initialization.")
        else:
            logger.info(
                f"Initialized with file-reading tool: {self.file_read_tool.name} - {self.file_read_tool.description}"
            )

    async def proofread_async(self, user_input):
        intent_data = self.inferrer.infer_intent_and_file(user_input)
        if intent_data.get("intent") == "proofread" and "file" in intent_data:
            file = intent_data["file"]
            if not self.file_read_tool:
                return "No tool available to read files."

            if self.use_stdio:
                server_params = StdioServerParameters(
                    command="python", args=[self.tool_selector.script_path], env=None
                )
                async with stdio_client(server_params) as streams:
                    async with ClientSession(*streams) as session:
                        await session.initialize()
                        response = await session.call_tool(self.file_read_tool.name, {"file": file})
                        if response.isError:
                            logger.error(f"Error calling Stdio tool: {response.content[0].text}")
                            return f"Error: {response.content[0].text}"
                        else:
                            logger.info(f"Stdio response: {response}")
                            file_content = response.content[0].text
                            proofread_result = self.proofreader.proofread(file_content)
                            return proofread_result
            else:
                async with sse_client(url=self.tool_selector.mcp_uri) as streams:
                    async with ClientSession(*streams) as session:
                        await session.initialize()
                        response = await session.call_tool(self.file_read_tool.name, {"file": file})
                        if response.isError:
                            logger.error(f"Error calling SSE tool: {response.content[0].text}")
                            return f"Error: {response.content[0].text}"
                        else:
                            logger.info(f"SSE response: {response}")
                            file_content = response.content[0].text
                            proofread_result = self.proofreader.proofread(file_content)
                            return proofread_result

        return "Sorry, I couldn't understand your request. Try asking to proofread a file, e.g., 'Check example.txt'."

    def proofread(self, user_input):
        return asyncio.run(self.proofread_async(user_input))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python proofread_agent_mcp.py <filename> [--stdio | --sse URI]")
        sys.exit(1)

    filename = sys.argv[1]
    user_input = f"Please proofread {filename}"

    # Parse optional arguments
    script_path = None
    mcp_uri = None
    if "--stdio" in sys.argv:
        script_path = (
            sys.argv[sys.argv.index("--stdio") + 1]
            if sys.argv.index("--stdio") + 1 < len(sys.argv)
            else "filesystem.py"
        )
    elif "--sse" in sys.argv:
        mcp_uri = (
            sys.argv[sys.argv.index("--sse") + 1]
            if sys.argv.index("--sse") + 1 < len(sys.argv)
            else "http://127.0.0.1:8000/sse"
        )

    agent = ProofReadAgentMCP(script_path=script_path, mcp_uri=mcp_uri)
    response = agent.proofread(user_input)
    print(response)
