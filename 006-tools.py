from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

from langchain.agents import create_agent

weather_agent = create_agent(
    model="gpt-4o-mini",
    tools=[web_search]
)

from langchain.messages import HumanMessage

question = HumanMessage(content="How is the weather today (Jan 3rd, 2026) in San Francisco?")

response = weather_agent.invoke(
    {"messages": [question]}
)

print(response['messages'][-1].content)