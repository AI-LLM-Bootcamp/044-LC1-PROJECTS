from dotenv import load_dotenv
# load_dotenv() reads the .env file in the project folder.
# It must contain a line like: OPENAI_API_KEY=sk-...
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from pprint import pprint
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

async def main():
    client = MultiServerMCPClient(
        {
            "time": {
                "transport": "stdio",
                # the next lines are pasted from the website of the MCP provider
                # see https://mcp.so/server/time/modelcontextprotocol
                # uvx comes with uv (https://docs.astral.sh/uv/). If you get
                # "No such file or directory: 'uvx'", install uv first,
                # for example with: pipx install uv
                "command": "uvx",
                "args": [
                    "mcp-server-time",
                    "--local-timezone=America/New_York"
                ]
            }
        }
    )

    tools = await client.get_tools()

    # This agent uses tools (the ones from the MCP server). Through the Chat
    # Completions endpoint, which LangChain uses by default, gpt-6-luna only
    # accepts tools when reasoning is turned off, so we create the model with
    # reasoning_effort="none". The model then answers directly, like gpt-4o-mini did.
    model = init_chat_model(MODEL, reasoning_effort="none")

    agent = create_agent(
        model=model,
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
