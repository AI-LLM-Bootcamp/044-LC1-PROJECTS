from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project folder.
# It must contain a line like: OPENAI_API_KEY=sk-...
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2


# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

# All the agents below use tools. Through the Chat Completions endpoint,
# which LangChain uses by default, gpt-6-luna only accepts tools when
# reasoning is turned off, so we create the model with reasoning_effort="none"
# and share it between the three agents. The model then answers directly,
# like gpt-4o-mini did.
model = init_chat_model(MODEL, reasoning_effort="none")

# create subagents

subagent_1 = create_agent(
    model=model,
    tools=[square_root]
)

subagent_2 = create_agent(
    model=model,
    tools=[square]
)

from langchain.messages import HumanMessage

@tool
def call_subagent_1(x: float) -> str:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = subagent_1.invoke({"messages": [HumanMessage(content=f"Calculate the square root of {x}")]})
    return response["messages"][-1].content

@tool
def call_subagent_2(x: float) -> str:
    """Call subagent 2 in order to calculate the square of a number"""
    response = subagent_2.invoke({"messages": [HumanMessage(content=f"Calculate the square of {x}")]})
    return response["messages"][-1].content

## Creating the main agent

main_agent = create_agent(
    model=model,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt="You are a helpful assistant who can call subagents to calculate the square root or square of a number.")


question = "What is the square root of 456?"

response = main_agent.invoke({"messages": [HumanMessage(content=question)]})

print(response['messages'][-1].content)
