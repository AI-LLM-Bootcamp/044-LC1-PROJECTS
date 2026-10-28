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

results = vector_store.similarity_search(
    "According to Gartner, what percentage of enterprises will use Generative AI APis or deploy generative AI-enabled applications in production environments in 2026?"
)

print(results[0])
