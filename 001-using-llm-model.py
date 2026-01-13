from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="gpt-4o-mini",
    temperature=0.0
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