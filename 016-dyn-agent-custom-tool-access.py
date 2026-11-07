from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project folder.
# It must contain the lines OPENAI_API_KEY=sk-... and TAVILY_API_KEY=tvly-...
load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain_community.utilities import SQLDatabase

# Fix: Specify the tables with correct capitalization to avoid error
db = SQLDatabase.from_uri(
    "sqlite:///Chinook.db",
    include_tables=['Artist', 'Album', 'Track', 'Customer', 'Invoice']  # Use exact names!
)

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

@tool
def sql_query(query: str) -> str:

    """Obtain information from the database using SQL queries"""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

from dataclasses import dataclass

@dataclass
class UserRole:
    user_role: str = "external"

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def dynamic_tool_call(request: ModelRequest, 
handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:

    """Dynamically call tools based on the runtime context"""

    user_role = request.runtime.context.user_role
    
    if user_role == "internal":
        pass # internal users get access to all tools
    else:
        tools = [web_search] # external users only get access to web search
        request = request.override(tools=tools) 

    return handler(request)

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

# This agent uses tools (web_search and sql_query). Through the Chat
# Completions endpoint, which LangChain uses by default, gpt-6-luna only
# accepts tools when reasoning is turned off, so we set reasoning_effort="none".
# The model then answers directly, like gpt-4o-mini did. (The other option
# OpenAI offers is the Responses API: init_chat_model(MODEL, use_responses_api=True).)
# We create the model object here, instead of passing its name as text,
# so that we can add this parameter.
model = init_chat_model(MODEL, reasoning_effort="none")

agent = create_agent(
    model=model,
    tools=[web_search, sql_query],
    middleware=[dynamic_tool_call],
    context_schema=UserRole
)

from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="How many artists are in the database?")]},
    context={"user_role": "external"}
)

# Recent OpenAI models mark the sources they use with citation markers,
# for example "cite turn6f4581" wrapped in invisible characters.
# Apps like ChatGPT turn them into links. Here we just remove them,
# as OpenAI's citation formatting guide suggests when you don't show links.
import re

def remove_citations(text):
    """Remove OpenAI citation markers from the model's answer."""
    return re.sub(r"\ue200cite\ue202.*?\ue201", "", text).strip()

print(remove_citations(response["messages"][-1].content))