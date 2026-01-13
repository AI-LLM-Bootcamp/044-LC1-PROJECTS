from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pydantic import BaseModel

class ArticleFormat(BaseModel):
    title: str
    subtitle: str
    body: str

agent = create_agent(
    model='gpt-4o-mini',
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