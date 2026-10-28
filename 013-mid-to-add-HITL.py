from dotenv import load_dotenv

# Remember: the .env file in the project folder must have the line
# OPENAI_API_KEY=sk-... (no quotes, no spaces).
load_dotenv()

"""
Human-in-the-Loop Email Agent - Terminal Version
Run this script from your terminal: python hitl_email_agent.py
"""

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langgraph.types import Command
from pprint import pprint

# Tool definitions
@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email from the given address."""
    return runtime.state["email"]

@tool
def send_email(body: str) -> str:
    """Send an email to the given address with the given subject."""
    return f"✅ Email sent successfully!\n\nBody: {body}"

# State and agent setup
class EmailState(AgentState):
    email: str

# Original version: this example used gpt-4o-mini.
# OpenAI has deprecated it (shutdown on January 14, 2027) and recommends
# gpt-6-luna as its replacement: small, fast and affordable for practice.
MODEL = "gpt-6-luna"

# This agent uses tools (read_email and send_email). Through the Chat
# Completions endpoint, which LangChain uses by default, gpt-6-luna only
# accepts tools when reasoning is turned off, so we set reasoning_effort="none".
# The model then answers directly, like gpt-4o-mini did. (The other option
# OpenAI offers is the Responses API: init_chat_model(MODEL, use_responses_api=True).)
# We create the model object here, instead of passing its name as text,
# so that we can add this parameter.
model = init_chat_model(MODEL, reasoning_effort="none")

agent = create_agent(
    model=model,
    tools=[read_email, send_email],
    state_schema=EmailState,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_email": False,
                "send_email": True,
            },
            description_prefix="Tool execution requires approval",
        ),
    ],
)

# Configuration
config = {"configurable": {"thread_id": "1"}}

# Helper function to print section headers
def print_header(text, symbol="="):
    """Print a formatted header"""
    print("\n" + symbol * 70)
    print(f"  {text}")
    print(symbol * 70 + "\n")

def print_subheader(text):
    """Print a formatted subheader"""
    print("\n" + "-" * 70)
    print(f"  {text}")
    print("-" * 70)

# Function to handle interrupts
def handle_interrupt(response, config, interrupt_number=1):
    """Handle a single interrupt and return the updated response"""
    
    if '__interrupt__' not in response:
        return response, False  # No interrupt, we're done
    
    print_header(f"⚠️  INTERRUPT #{interrupt_number} - Human approval required!", "!")
    
    # Extract interrupt info
    interrupt_obj = response['__interrupt__'][0]
    interrupt_value = interrupt_obj.value
    
    # Extract tool information from the interrupt value
    action_requests = interrupt_value.get('action_requests', [])
    if action_requests:
        action_request = action_requests[0]
        tool_name = action_request.get('name', 'unknown')
        tool_args = action_request.get('args', {})
        description = action_request.get('description', '')
        
        print(f"🤖 Agent wants to call: {tool_name}")
        print(f"\n📝 With arguments:")
        pprint(tool_args)
        
        if description:
            print(f"\n📄 Description:")
            print(description)
    else:
        print("⚠️  No action requests found in interrupt")
        return response, False
    
    # Get human input
    print_subheader(f"🤔 DECISION TIME (Interrupt #{interrupt_number})")
    print("\nWhat do you want to do?")
    print("  1 - Approve: Let the agent execute as planned")
    print("  2 - Reject: Stop the action and provide feedback")
    print("  3 - Edit: Modify the content before executing")
    
    choice = input("\n👉 Enter your choice (1/2/3): ").strip()
    
    # Resume based on decision
    print_subheader(f"⚡ RESUMING (Interrupt #{interrupt_number})...")
    
    if choice == "1":
        # APPROVE
        print("\n✅ You APPROVED the action\n")
        new_response = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )
        return new_response, True
        
    elif choice == "2":
        # REJECT
        print("\n❌ You REJECTED the action")
        feedback = input("👉 Enter your feedback message: ").strip()
        print()
        new_response = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": "reject",
                            "message": feedback
                        }
                    ]
                }
            ),
            config=config
        )
        return new_response, True
        
    elif choice == "3":
        # EDIT
        print("\n✏️  You chose to EDIT the action")
        new_body = input("👉 Enter the new email body: ").strip()
        print()
        new_response = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": tool_name,
                                "args": {"body": new_body}
                            }
                        }
                    ]
                }
            ),
            config=config
        )
        return new_response, True
        
    else:
        print("\n❌ Invalid choice\n")
        return response, False


def main():
    """Main function to run the HITL agent"""
    
    # PHASE 1: Initial invocation
    print_header("🚀 PHASE 1: Starting Human-in-the-Loop Agent")
    
    print("Sending initial message to agent...")
    print("Email content: 'Hi, Donald. I will be in town tomorrow, will you have ")
    print("an available room in the White House? Best, Julio.'\n")
    
    response = agent.invoke(
        {
            "messages": [HumanMessage(content="Please read my email and send a response.")],
            "email": "Hi, Donald. I will be in town tomorrow, will you have an available room in the White House? Best, Julio."
        },
        config=config
    )
    
    # Handle all interrupts in a loop
    interrupt_count = 0
    max_interrupts = 5  # Safety limit to prevent infinite loops
    
    while '__interrupt__' in response and interrupt_count < max_interrupts:
        interrupt_count += 1
        response, continued = handle_interrupt(response, config, interrupt_count)
        
        if not continued:
            break
    
    # FINAL RESULT
    print_header("🎉 FINAL RESULT")
    
    if '__interrupt__' in response:
        print("⚠️  Still has pending interrupts (reached max limit or invalid choice)\n")
        print("Interrupt details:")
        pprint(response['__interrupt__'])
    else:
        print("✅ Process completed successfully!\n")
        
        # Show only the final messages
        if 'messages' in response:
            print("📧 Final Messages:")
            print("-" * 70)
            for msg in response['messages'][-3:]:  # Show last 3 messages
                msg_type = msg.type.upper() if hasattr(msg, 'type') else 'MESSAGE'
                msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                print(f"\n[{msg_type}]")
                print(msg_content)
                print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
