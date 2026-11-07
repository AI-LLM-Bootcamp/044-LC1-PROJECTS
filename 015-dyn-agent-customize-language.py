from dotenv import load_dotenv

# Remember: the .env file in the project folder must have the line
# OPENAI_API_KEY=sk-... (no quotes, no spaces).
load_dotenv()

from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class LanguageContext:
    user_language: str = "English"

@dynamic_prompt
def user_language_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_language = request.runtime.context.user_language
    base_prompt = "You are a helpful assistant."

    if user_language != "English":
        return f"{base_prompt} only respond in {user_language}."
    elif user_language == "English":
        return base_prompt

from langchain.agents import create_agent

# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
# This agent has no tools, so we only need to change the model name:
# create_agent still accepts it as text.
MODEL = "gpt-6-luna"

agent = create_agent(
    model=MODEL,
    context_schema=LanguageContext,
    middleware=[user_language_prompt]
)

from langchain.messages import HumanMessage

# The input key is "messages" (plural), the name LangChain uses for the
# conversation. With "message", the agent ignores the key without an error,
# so the model would only see the system prompt and not the user's question.
response = agent.invoke(
    {"messages": [HumanMessage(content="Hola, ¿cómo estás?")]},
    context=LanguageContext(user_language="Spanish")
)

print(response["messages"][-1].content)