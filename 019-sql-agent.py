from dotenv import load_dotenv

# Remember: the .env file in the project folder must have the line
# OPENAI_API_KEY=sk-... (no quotes, no spaces).
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
from langchain.chat_models import init_chat_model

# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

# This agent uses a tool (sql_query). Through the Chat Completions endpoint,
# which LangChain uses by default, gpt-6-luna only accepts tools when
# reasoning is turned off, so we set reasoning_effort="none". The model then
# answers directly, like gpt-4o-mini did. (The other option OpenAI offers is
# the Responses API: init_chat_model(MODEL, use_responses_api=True).)
# We create the model object here, instead of passing its name as text,
# so that we can add this parameter.
model = init_chat_model(MODEL, reasoning_effort="none")

agent = create_agent(
    model=model,
    tools=[sql_query]
)

from langchain.messages import HumanMessage

question = HumanMessage(content="How many artists are in the database?")

response = agent.invoke(
    {"messages": [question]}
)

print(response["messages"][-1].content)
