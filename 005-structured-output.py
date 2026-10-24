from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project folder.
# It must contain a line like: OPENAI_API_KEY=sk-...
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from pydantic import BaseModel

class ArticleFormat(BaseModel):
    title: str
    subtitle: str
    body: str

# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

# To get the structured response, the agent may ask the model to fill in
# ArticleFormat through a tool. Through the Chat Completions endpoint, which
# LangChain uses by default, gpt-6-luna only accepts tools when reasoning
# is turned off, so we create the model with reasoning_effort="none".
# The model then answers directly, like gpt-4o-mini did.
model = init_chat_model(MODEL, reasoning_effort="none")

agent = create_agent(
    model=model,
    system_prompt="You are an investigative journalist.",
    response_format=ArticleFormat
)

question = HumanMessage(content="Write a short article explaining briefly the top conspiracy theories about who killed JFK?")

response = agent.invoke(
    {"messages": [question]}
)

article = response["structured_response"]
article_title = article.title
article_subtitle = article.subtitle
article_body = article.body

print("\n=========\n")

print(f"The journalist wrote an article called {article_title}")

print("\n=========\n")

print(f"The article was about {article_subtitle}")

print("\n=========\n")

print(f"This is the body of the article:\n {article_body}")
