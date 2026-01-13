from dotenv import load_dotenv

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

agent = create_agent(
    model="gpt-4o-mini",
    tools=[web_search, sql_query],
    middleware=[dynamic_tool_call],
    context_schema=UserRole
)

from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="How many artists are in the database?")]},
    context={"user_role": "external"}
)

print(response["messages"][-1].content)