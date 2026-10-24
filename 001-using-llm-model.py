from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project folder.
# It must contain a line like: OPENAI_API_KEY=sk-...
load_dotenv()

from langchain.chat_models import init_chat_model

# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

# The original example used temperature=0.0 to get more stable answers.
# Recent OpenAI models are reasoning models and only accept the default
# value (1), so we set temperature=1 explicitly.
model = init_chat_model(
    model=MODEL,
    temperature=1
)


print("Who killed JFK?\n")

response = model.invoke("Who killed JFK?")
print(response.content)

print("\n=========\n")

print("What is the best coffee shop in San Francisco?\n")

response = model.invoke("What is the best coffee shop in San Francisco?")
print(response.content)

print("\n=========\n")

print("What is the best Spanish restaurant in San Francisco?\n")

for chunk in model.stream("What is the best Spanish restaurant in San Francisco?"):
    print(chunk.text, end="|", flush=True)
    
print("\n=========\n")

print("What is the best Meetup for European Expats in San Francisco?\n")

for chunk in model.stream("What is the best Meetup for European Expats in San Francisco?"):
    print(chunk.text, end="", flush=True)
