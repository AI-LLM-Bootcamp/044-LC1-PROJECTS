from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

async def main():
    client = MultiServerMCPClient(
        {
            "time": {
                "transport": "stdio",
                # the next lines are pasted from the website of the MCP provider
                # see https://mcp.so/server/time/modelcontextprotocol
                "command": "uvx",
                "args": [
                    "mcp-server-time",
                    "--local-timezone=America/New_York"
                ]
            }
        }
    )

    tools = await client.get_tools()

    agent = create_agent(
        model="gpt-4o-mini",
        tools=tools,
    )

    question = HumanMessage(content="What time is it in Madrid?")

    response = await agent.ainvoke(
        {"messages": [question]}
    )

    print(response['messages'][-1].content)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())