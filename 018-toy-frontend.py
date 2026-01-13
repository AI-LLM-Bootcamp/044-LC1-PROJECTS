"""
Simple LangChain 1.0 Agent for Learning
This agent can do math, tell jokes, and have conversations.
Updated to use the new create_agent function from LangChain 1.0
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Load environment variables from .env file
load_dotenv()

# Check if API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

# =============================================================================
# STEP 1: Define Tools (Functions your agent can use)
# =============================================================================

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The product of a and b
    """
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b
    """
    return a + b

@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers.
    
    Args:
        a: Numerator (number to be divided)
        b: Denominator (number to divide by)
    
    Returns:
        The result of a divided by b
    """
    if b == 0:
        return "Error: Cannot divide by zero!"
    return a / b

@tool
def tell_joke() -> str:
    """Tell a programming joke.
    
    Returns:
        A funny programming joke
    """
    jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "Why do Java developers wear glasses? Because they don't C#!",
        "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
        "Why did the programmer quit his job? Because he didn't get arrays!",
    ]
    import random
    return random.choice(jokes)

# =============================================================================
# STEP 2: Create the Language Model
# =============================================================================

# Using GPT-4o-mini because it's cost-effective for learning
# You can also use: "gpt-4o", "gpt-3.5-turbo", etc.
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0  # Controls randomness (0=focused, 1=creative)
)

# =============================================================================
# STEP 3: Combine Tools into a List
# =============================================================================

tools = [multiply, add, divide, tell_joke]

# =============================================================================
# STEP 4: Create the Agent (NEW LANGCHAIN 1.0 WAY)
# =============================================================================

# create_agent is the new standard in LangChain 1.0
# - Replaces the deprecated create_react_agent from langgraph.prebuilt
# - Uses system_prompt instead of state_modifier (clearer naming)
# - Still builds on LangGraph runtime under the hood
# - Handles the message flow automatically
# - Creates the proper state structure (with the 'messages' key)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful assistant that can do math and tell jokes. "
                  "Always be friendly and clear in your responses."
)

# =============================================================================
# WHAT'S HAPPENING UNDER THE HOOD
# =============================================================================
# The 'agent' variable now contains a complete LangGraph compiled graph:
# 
# 1. It has a STATE that tracks the conversation (messages list)
# 2. It has NODES:
#    - "agent" node: Where the LLM thinks and decides what to do
#    - "tools" node: Where tools actually execute
# 3. It has EDGES connecting these nodes
# 4. It follows the ReAct pattern: Reason → Act → Observe (repeat)
#
# Agent Chat UI expects this specific structure with the 'messages' key!
# The create_agent function ensures this contract is met automatically.
# =============================================================================

# This allows the agent to be imported by other files
if __name__ == "__main__":
    print("Agent is ready! Use 'langgraph dev' to start the server.")