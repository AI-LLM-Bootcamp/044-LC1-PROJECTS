from dotenv import load_dotenv

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

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

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

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_handbook],
    system_prompt="You are a helpful agent that can search the pdf file for information."
    )

from langchain.messages import HumanMessage

response = agent.invoke(
    {"messages": [HumanMessage(content="According to Gartner, what percentage of enterprises will use Generative AI APis or deploy generative AI-enabled applications in production environments in 2026?")]}
)

print(response["messages"][-1].content)