from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

# --- 1. Define Tools (ADK Style) ---
def remember_session_name(name: str, tool_context: ToolContext) -> str:
    """Remembers the user's name for this session."""
    # NOTE: No 'user:' prefix. This is session-scoped.
    tool_context.state["session_name"] = name
    return f"OK, I'll remember you as {name} for this session."

# --- 2. Define the Agent ---
# We don't need a 'get_name' tool. The agent can read state.
root_agent = Agent(
    name="StatefulAgent",
    model="gemini-2.5-flash",
    description="A simple agent that remembers your name during the session.",

    instruction=(
        "You are a helpful assistant. You can remember the user's name "
        "for the current session. "
        "If the user tells you their name, use `remember_session_name`. "
        "If the user asks for their name, check the state for 'session_name' "
        "and tell them. Otherwise, say you don't know."
    ),
    tools=[remember_session_name],
)
