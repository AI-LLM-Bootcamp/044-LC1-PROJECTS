from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o-mini",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 100),
            keep=("messages", 1)
        )
    ],
)

from langchain.messages import HumanMessage, AIMessage

response = agent.invoke(
    {"messages": [
        HumanMessage(content="Are you ready to play the JFK QA game?"),
        AIMessage(content="Sure!"),
        HumanMessage(content="Who was the favorite sister or JFK?"),
        AIMessage(content="Her sister Kick."),
        HumanMessage(content="Correct! Who was his favorite brother?"),
        AIMessage(content="Hmmm, that is difficult. I would say Ted. He loved Bobby very much, but Bobby was very different from him."),
        HumanMessage(content="Correct! What was the main source of pain of JFK on a daily basis?"),
        AIMessage(content="Back pain."),
        HumanMessage(content="Correct again! Did JFK have dogs?"),
        ]},
    {"configurable": {"thread_id": "1"}}
)

print(response["messages"][0].content)