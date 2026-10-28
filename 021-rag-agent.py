from dotenv import load_dotenv

# Remember: the .env file in the project folder must have the line
# OPENAI_API_KEY=sk-... (no quotes, no spaces).
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("gen-ai-in-2026.pdf")

data = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(data)

from langchain_openai import OpenAIEmbeddings

# text-embedding-3-large is still one of OpenAI's current embedding models,
# with no shutdown date announced, so this example keeps it.
# Writing the model name once makes it easy to see and change.
EMBEDDING_MODEL = "text-embedding-3-large"

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

from langchain.tools import tool

@tool
def search_handbook(query: str) -> str:
    """Search the handbook for information"""
    results = vector_store.similarity_search(query)
    return results[0].page_content

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

# This agent uses a tool (search_handbook). Through the Chat Completions
# endpoint, which LangChain uses by default, gpt-6-luna only accepts tools
# when reasoning is turned off, so we set reasoning_effort="none". The model
# then answers directly, like gpt-4o-mini did. (The other option OpenAI offers
# is the Responses API: init_chat_model(MODEL, use_responses_api=True).)
# We create the model object here, instead of passing its name as text,
# so that we can add this parameter.
model = init_chat_model(MODEL, reasoning_effort="none")

agent = create_agent(
    model=model,
    tools=[search_handbook],
    system_prompt="You are a helpful agent that can search the pdf file for information."
    )

from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="According to Gartner, what percentage of enterprises will use Generative AI APis or deploy generative AI-enabled applications in production environments in 2026?")]}
)

# Recent OpenAI models may mark the sources they use with citation markers,
# for example "cite turn0search0" wrapped in invisible characters.
# Apps like ChatGPT turn them into links. Here we just remove them,
# as OpenAI's citation formatting guide suggests when you don't show links.
import re

def remove_citations(text):
    """Remove OpenAI citation markers from the model's answer."""
    return re.sub(r"cite.*?", "", text).strip()

print(remove_citations(response["messages"][-1].content))
