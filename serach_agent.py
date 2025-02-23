import asyncio
import logging
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchAgent:
    def __init__(self, api_key):
        self.server_config = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": api_key},
        }
        self.use_stdio = True

    async def start_server(self):
        server_params = StdioServerParameters(
            command=self.server_config["command"],
            args=self.server_config["args"],
            env=self.server_config["env"],
        )
        return stdio_client(server_params)

    async def search(self, query):
        async with await self.start_server() as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Assuming the MCP tool for search is named 'search_query' and accepts 'query' parameter
                response = await session.call_tool("search_query", {"query": query})
                if response.isError:
                    logger.error(f"Error calling search tool: {response.content[0].text}")
                    return f"Error: {response.content[0].text}"
                else:
                    logger.info(f"Search response: {response}")
                    return response.content[0].text

    async def run(self, query):
        try:
            result = await self.search(query)
            print(result)
        except Exception as e:
            logger.error(f"An error occurred: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python search_agent.py <query>")
        sys.exit(1)

    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        print("Error: BRAVE_API_KEY environment variable not set")
        sys.exit(1)
    query = sys.argv[1]

    agent = SearchAgent(api_key)
    asyncio.run(agent.run(query))


if __name__ == "__main__":
    main()
