from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent

from langgraph.checkpoint.memory import InMemorySaver

# This is where we add the short-term memory ability to our agent
agent2 = create_agent(
    "gpt-4o-mini",
    checkpointer=InMemorySaver(),  
)

from langchain.messages import HumanMessage

question = HumanMessage(content="Hello my name is Julio and I like vespas.")

# This is where we set the conversation ID
config = {"configurable": {"thread_id": "1"}}

# This is where we associate our conversation with the conversation ID
response = agent2.invoke(
    {"messages": [question]},
    config,  
)

print("\n=========\n")

print(response['messages'][-1].content)

question = HumanMessage(content="What is my name? What is my favorite scooter?")

response = agent2.invoke(
    {"messages": [question]},
    config,  
)

print("\n=========\n")

print(response['messages'][-1].content)

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import AgentState

# This is where we set the format (aka schema) of our custom shor-term memory (aka State)
class CustomAgentState(AgentState):  
    user_id: str
    user_preferences: dict

# This is where we add the short-term memory ability to our agent
agent3 = create_agent(
    "gpt-4o-mini",
    state_schema=CustomAgentState, 
    checkpointer=InMemorySaver(),  
)

response = agent3.invoke(
    {
        "messages": [{
            "role": "user", 
            "content": "My favorite city is San Francisco."}],
        "user_id": "user_123",  # This is just for demo, we are not using this
        "user_preferences": {"converation_style": "Casual"}  # This is just for demo, we are not using this
    },
    {"configurable": {"thread_id": "1"}})


print("\n=========\n")

print(response['messages'][-1].content)

response = agent3.invoke(
    {
        "messages": [{
            "role": "user", 
            "content": "Do you know my favorite city? And my preferred conversation style?"}],
        "user_id": "user_123",  
        "user_preferences": {"converation_style": "Casual"}  
    },
    {"configurable": {"thread_id": "1"}})

print("\n=========\n")

print(response['messages'][-1].content)

print("\n=========\n")

print(response['user_preferences'])