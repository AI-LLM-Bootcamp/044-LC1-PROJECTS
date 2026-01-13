from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent

system_prompt = "You are an investigative journalist."

agent = create_agent(
    model="gpt-4o-mini",
    system_prompt=system_prompt
)

from langchain.messages import HumanMessage

print("\n=========\n")

print("Who really killed JFK?\n")

response = agent.invoke(
    {"messages": [HumanMessage(content="Who really killed JFK?")]}
)

print(response['messages'][-1].content)

print("\n=========\n")

print("Really, who really killed JFK?\n")


for token, metadata in agent.stream(
    {"messages": [HumanMessage(content="Really, who really killed JFK?")]},
    stream_mode="messages"
):
    
    if token.content:
        print(token.content, end="", flush=True)