"""
✅ REAL MCP Client using the official `mcp` Python library.

Spawns the MCP server as a subprocess and communicates via stdio transport
using the actual Model Context Protocol — not a fake HTTP/JSON class.

Used by the agent to call tools like:
    await mcp_client.call_tool("get_doctors_by_symptoms", {"symptoms": [...]})
"""
import asyncio
import json
import logging
import sys
import os
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

# ✅ Real MCP library imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Path to the MCP server module
MCP_SERVER_MODULE = "app.mcp.mcp_server"


class MCPClientSession:
    """
    Real MCP Client that:
    1. Spawns the MCP server as a subprocess (stdio transport)
    2. Uses the official MCP ClientSession to call tools
    3. Returns structured results to the agent
    """

    def __init__(self, server_params: dict):
        # Convert dict to StdioServerParameters
        self._server_params = StdioServerParameters(
            command=server_params.get("command"),
            args=server_params.get("args", []),
            env=server_params.get("env", None),
        )
        logger.info(f"MCP server params: {self._server_params}")  # Add this for debugging

    @asynccontextmanager
    async def _get_session(self):
        """Context manager that yields a live MCP ClientSession."""
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all tools available on the MCP server."""
        try:
            async with self._get_session() as session:
                result = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    }
                    for tool in result.tools
                ]
        except Exception as e:
            logger.error(f"MCP list_tools failed: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: dict):
        """
        Call a tool on the MCP server via real MCP protocol.

        Args:
            tool_name: Name of the MCP tool (e.g. "get_doctors_by_symptoms")
            arguments: Tool input arguments dict

        Returns:
            Parsed dict result from the tool
        """
        logger.info(f"[MCP Client] Calling tool: {tool_name} | args: {str(arguments)[:150]}")

        try:
            async with self._get_session() as session:
                result = await session.call_tool(tool_name, arguments=arguments)

                # MCP returns content blocks — extract the text
                raw_text = ""
                for block in result.content:
                    if hasattr(block, "text"):
                        raw_text += block.text

                # Our tools always return JSON strings
                parsed = json.loads(raw_text) if raw_text else {}
                logger.info(f"[MCP Client] Tool {tool_name} result: {str(parsed)[:200]}")
                return parsed

        except Exception as e:
            logger.error(f"[MCP Client] Tool call '{tool_name}' failed: {e}", exc_info=True)
            # Optionally, return a default response or re-raise with more context
            raise HTTPException(status_code=500, detail=f"MCP tool call failed: {str(e)}")

    def get_tools_for_llm_prompt(self) -> str:
        """
        Synchronously fetch tool descriptions for embedding in LLM system prompt.
        Runs a short event loop to list tools.
        """
        try:
            loop = asyncio.new_event_loop()
            tools = loop.run_until_complete(self.list_tools())
            loop.close()
        except Exception:
            tools = []

        if not tools:
            return "(MCP tools unavailable — check mcp_server is runnable)"

        lines = []
        for tool in tools:
            schema = tool.get("inputSchema", {})
            props = schema.get("properties", {})
            required = schema.get("required", [])
            param_lines = [
                f"    - {k} ({'required' if k in required else 'optional'}): "
                f"{v.get('description', v.get('type', ''))}"
                for k, v in props.items()
            ]
            lines.append(
                f"Tool: {tool['name']}\n"
                f"Description: {tool['description']}\n"
                f"Parameters:\n" + "\n".join(param_lines)
            )

        return "\n\n".join(lines)


# Singleton — import this in the agent
mcp_client = MCPClientSession(server_params={
    "command": sys.executable,
    "args": ["-m", MCP_SERVER_MODULE],
    "env": {**os.environ},
})