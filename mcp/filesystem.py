import asyncio
from mcp.server import FastMCP
import os

mcp = FastMCP("ta-filesystem")


@mcp.tool()
async def read_file(file: str) -> str:
    """
    Read the content of a file.
    Args:
        file: The name of the file to read.
    """
    try:
        with open(file, "r") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file}' not found")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")


async def start_server():
    await mcp.run_sse_async()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(start_server())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
