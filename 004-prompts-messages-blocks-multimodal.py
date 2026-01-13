from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
import base64
import os

model = init_chat_model("gpt-4o-mini")

# Example 1: Simple text prompt
print("\n" + "="*60)
print("Example 1: Simple Text Prompt")
print("="*60)
response = model.invoke("What was the main physical problem of JFK?")
print(response.content)

# Example 2: Messages with SystemMessage and HumanMessage objects
print("\n" + "="*60)
print("Example 2: Using Message Objects")
print("="*60)
messages = [
    SystemMessage("You are an expert on San Francisco"),
    HumanMessage("What is the best coffee shop in the city?"),
]
response = model.invoke(messages)
print(f"Role: {response.type}")
print(f"Content: {response.content}")

# Example 3: Messages with dictionary format
print("\n" + "="*60)
print("Example 3: Using Dictionary Format")
print("="*60)
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about coffee"}
]
response = model.invoke(messages)
print(f"Role: {response.type}")
print(f"Content: {response.content}")

# Example 4: Message with metadata
print("\n" + "="*60)
print("Example 4: Message with Metadata")
print("="*60)
human_msg = HumanMessage(
    content="Hello! Tell me a fun fact about the Kennedy family.",
    name="bobby",
    id="msg_123",
)
response = model.invoke([human_msg])
print(f"User: {human_msg.name}")
print(f"Message ID: {human_msg.id}")
print(f"Response: {response.content}")

# Example 5: Message Content and Content Blocks with LOCAL image
print("\n" + "="*60)
print("Example 5: Message Content and Content Blocks")
print("="*60)

# String content
human_message = HumanMessage("Hello, how are you?")
print("Simple text message:")
print(human_message.content)

print("\n---\n")

# Load local image and encode to base64
image_path = "jackie.jpg"  # or "image.png", etc.

if os.path.exists(image_path):
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Determine image format
    image_extension = os.path.splitext(image_path)[1].lower()
    format_map = {
        '.jpg': 'jpeg',
        '.jpeg': 'jpeg',
        '.png': 'png',
        '.gif': 'gif',
        '.webp': 'webp'
    }
    image_format = format_map.get(image_extension, 'jpeg')
    
    # Provider-native (OpenAI-style) content blocks with local image
    human_message = HumanMessage(content=[
        {"type": "text", "text": "Describe this image in Spanish"},
        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}},
    ])
    
    # Print in a readable way (don't print the full base64!)
    print("OpenAI-style content blocks:")
    print(f"Text: {human_message.content[0]['text']}")
    print(f"Image: data:image/{image_format};base64,[{len(image_data)} characters of base64 data]")
    
    print("\n---\n")
    
    # LangChain 1.0 content blocks with local image
    human_message = HumanMessage(content=[
        {"type": "text", "text": "Describe this image in English"},
        {"type": "image", "url": f"data:image/{image_format};base64,{image_data}"},
    ])
    
    print("LangChain 1.0 content blocks:")
    print(f"Text: {human_message.content[0]['text']}")
    print(f"Image: data:image/{image_format};base64,[{len(image_data)} characters of base64 data]")
    
    print("\n---\n")
    
    # Actually INVOKE the model to see the response
    print("Model Response:")
    response = model.invoke([human_message])
    print(response.content)
    
else:
    print(f"Error: Image file '{image_path}' not found in project directory.")
    print("Please add an image file to your project root.")

# Example 6: Multi-modal message with LOCAL image file
print("\n" + "="*60)
print("Example 6: Multi-modal Message (Text + Local Image)")
print("="*60)

# Path to your local image (put an image.jpg in your project root)
image_path = "jackie.jpg"  # or "image.png", "photo.jpeg", etc.

# Check if file exists
if os.path.exists(image_path):
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Determine image format from file extension
    image_extension = os.path.splitext(image_path)[1].lower()
    format_map = {
        '.jpg': 'jpeg',
        '.jpeg': 'jpeg',
        '.png': 'png',
        '.gif': 'gif',
        '.webp': 'webp'
    }
    image_format = format_map.get(image_extension, 'jpeg')
    
    # Create the message
    human_message = HumanMessage(content=[
        {"type": "text", "text": "What do you see in this image? Describe it in detail. Can you recognize who is this very famous person?"},
        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}}
    ])
    
    response = model.invoke([human_message])
    print(f"Image file: {image_path}")
    print(f"Question: What do you see in this image?")
    print(f"Response: {response.content}")
else:
    print(f"Note: Image file '{image_path}' not found in project directory.")
    print("Please add an image file (e.g., image.jpg, photo.png) to your project root.")

# Example 7: Alternative format with dictionary (using local image)
print("\n" + "="*60)
print("Example 7: Image with Dictionary Format (Local File)")
print("="*60)

# You can use a different image file
image_path_2 = "jackie.jpg"  # or any other image in your project

if os.path.exists(image_path_2):
    with open(image_path_2, "rb") as image_file:
        image_data_2 = base64.b64encode(image_file.read()).decode('utf-8')
    
    image_extension_2 = os.path.splitext(image_path_2)[1].lower()
    image_format_2 = format_map.get(image_extension_2, 'jpeg')
    
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this image and tell me what's interesting about it. Can you recognize who is this very famous person?"},
            {"type": "image_url", "image_url": {"url": f"data:image/{image_format_2};base64,{image_data_2}"}},
        ]
    }
    response = model.invoke([message])
    print(f"Image file: {image_path_2}")
    print(f"Response: {response.content}")
else:
    print(f"Note: Image file '{image_path_2}' not found.")
    print("Using same image as Example 6 if it exists...")
    
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            image_data_2 = base64.b64encode(image_file.read()).decode('utf-8')
        
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image and tell me what's interesting about it."},
                {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data_2}"}},
            ]
        }
        response = model.invoke([message])
        print(f"Image file: {image_path}")
        print(f"Response: {response.content}")

print("\n" + "="*60)
print("All Examples Completed!")
print("="*60 + "\n")