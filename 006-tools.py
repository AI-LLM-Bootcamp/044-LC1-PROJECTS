from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project folder.
# It must contain the lines OPENAI_API_KEY=sk-... and TAVILY_API_KEY=tvly-...
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
from langchain.chat_models import init_chat_model

# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

# This agent uses tools. Through the Chat Completions endpoint, which
# LangChain uses by default, gpt-6-luna only accepts tools when reasoning
# is turned off, so we create the model with reasoning_effort="none".
# The model then answers directly, like gpt-4o-mini did.
model = init_chat_model(MODEL, reasoning_effort="none")

weather_agent = create_agent(
    model=model,
    tools=[web_search]
)

from langchain.messages import HumanMessage

question = HumanMessage(content="How is the weather today in San Francisco?")

response = weather_agent.invoke(
    {"messages": [question]}
)

# Recent OpenAI models mark the sources they use with citation markers,
# for example "cite turn6f4581" wrapped in invisible characters.
# Apps like ChatGPT turn them into links. Here we just remove them,
# as OpenAI's citation formatting guide suggests when you don't show links.
import re

def remove_citations(text):
    """Remove OpenAI citation markers from the model's answer."""
    return re.sub(r"\ue200cite\ue202.*?\ue201", "", text).strip()

print(remove_citations(response['messages'][-1].content))
