from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project folder.
# This example uses two providers, so it must contain the lines
# OPENAI_API_KEY=sk-... and ANTHROPIC_API_KEY=sk-ant-...
load_dotenv()

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

# Original version: this example used claude-sonnet-4-5 as the large model
# and gpt-4o-mini as the standard model. Both providers have announced
# their retirement:
# - Anthropic retires claude-sonnet-4-5 on November 30, 2026 and recommends
#   claude-sonnet-5-5 as its replacement.
# - OpenAI shuts down gpt-4o-mini on January 14, 2027 and recommends
#   gpt-6-luna as its replacement: small, fast and affordable for practice.
# We define each model once, so you can change it by editing one line.
# (We don't set temperature: recent Claude models no longer accept it.)
LARGE_MODEL = "claude-sonnet-5-5"
STANDARD_MODEL = "gpt-6-luna"

large_model = init_chat_model(LARGE_MODEL)
standard_model = init_chat_model(STANDARD_MODEL)


@wrap_model_call
def state_based_model(request: ModelRequest, 
handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Select model based on State conversation length."""
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)  

    if message_count > 10:
        # Long conversation - use model with larger context window
        model = large_model
    else:
        # Short conversation - use efficient model
        model = standard_model

    request = request.override(model=model)  

    return handler(request)

from langchain.agents import create_agent

agent = create_agent(
    model=STANDARD_MODEL,
    middleware=[state_based_model],
    system_prompt="You are roleplaying a real life helpful office intern."
)

from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [
        HumanMessage(content="Did you water the office plant today?")
        ]}
)

print(response["messages"][-1].content)

print(response["messages"][-1].response_metadata["model_name"])

print("\n")
print("========= End of the first respose. Now it starts the second response: ===========")
print("\n")

from langchain.messages import AIMessage

# The AIMessage entries below are written by hand to simulate an earlier
# conversation. They don't call any model, so they stay as in the original.

response = agent.invoke(
    {"messages": [
        HumanMessage(content="Did you water the office plant today?"),
        AIMessage(content="Yes, I gave it a light watering this morning."),
        HumanMessage(content="Has it grown much this week?"),
        AIMessage(content="It's sprouted two new leaves since Monday."),
        HumanMessage(content="Are the leaves still turning yellow on the edges?"),
        AIMessage(content="A little, but it's looking healthier overall."),
        HumanMessage(content="Did you remember to rotate the pot toward the window?"),
        AIMessage(content="I rotated it a quarter turn so it gets more even light."),
        HumanMessage(content="How often should we be fertilizing this plant?"),
        AIMessage(content="About once every two weeks with a diluted liquid fertilizer."),
        HumanMessage(content="When should we expect to have to replace the pot?")
        ]}
)

print(response["messages"][-1].content)

print(response["messages"][-1].response_metadata["model_name"])