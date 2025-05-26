#from smolagents import ToolCollection
#from mcp.client.stdio import StdioServerParameters
#
#server_parameters = StdioServerParameters(command="uv", args=["run", "server.py"])
#
#with ToolCollection.from_mcp(
#    server_parameters, trust_remote_code=True
#) as tools:
#    print("\n".join(f"{tool.name}: {tool.description}" for tool in tools.tools))

from smolagents.mcp_client import MCPClient

with MCPClient(
    {"url": "http://127.0.0.1:7860/gradio_api/mcp/sse"}
) as tools:
    # Tools from the remote server are available
    print("\n".join(f"{t.name}: {t.description}" for t in tools))

#from smolagents import InferenceClientModel, CodeAgent, ToolCollection
#from mcp.client.stdio import StdioServerParameters
#
#model = InferenceClientModel()
#
#server_parameters = StdioServerParameters(command="uv", args=["run", "server.py"])
#
#with ToolCollection.from_mcp(
#    server_parameters, trust_remote_code=True
#) as tool_collection:
#    agent = CodeAgent(tools=[*tool_collection.tools], model=model)
#    agent.run("What's the weather in Tokyo?")