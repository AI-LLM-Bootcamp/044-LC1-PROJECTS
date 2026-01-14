from dotenv import load_dotenv

load_dotenv()

from langchain_community.utilities import SQLDatabase

# Fix: Specify the tables with correct capitalization to avoid error
db = SQLDatabase.from_uri(
    "sqlite:///Chinook.db",
    include_tables=['Artist', 'Album', 'Track', 'Customer', 'Invoice']  # Use exact names!
)

from langchain.tools import tool

@tool
def sql_query(query: str) -> str:

    """Obtain information from the database using SQL queries"""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o-mini",
    tools=[sql_query]
)

from langchain.messages import HumanMessage

question = HumanMessage(content="How many artists are in the database?")

response = agent.invoke(
    {"messages": [question]}
)

print(response["messages"][-1].content)