from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project folder.
# It must contain a line like: OPENAI_API_KEY=sk-...
load_dotenv()

from langchain.agents import create_agent

# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

system_prompt = "You are an investigative journalist."

agent = create_agent(
    model=MODEL,
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
