import sqlite3
import os
import json
from typing import TypedDict, Annotated, List, Optional
from absl import app, flags
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from pathlib import Path

FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", False, "Enable debug logging.")

# Load environment variables
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# 1. Define the State
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], lambda x, y: x + y if y else x]
    user_name: Optional[str]

# 2. Define the Tools
@tool
def remember_name(name: str) -> str:
    """Use this tool to remember the user's name."""
    return json.dumps({"name_saved": name})

tools = [remember_name]
tool_executor = ToolNode(tools)

# Rationale for global initialization:
# Initializing the LLM and binding tools are expensive operations.
# By defining `llm` and `agent_runnable` here, we create a single,
# reusable instance that persists for the application's lifecycle.
# This avoids the severe performance overhead of re-initializing
# the model on every invocation of the `call_model` node.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
agent_runnable = llm.bind_tools(tools)

def get_system_prompt(name: Optional[str]) -> str:
    """Generates the system prompt based on whether the user's name is known."""
    system_prompt = "You are a helpful assistant."
    if name:
        system_prompt += f" The user's name is {name}."
    else:
        system_prompt += (
            " You do not know the user's name yet. "
            "When the user provides their name, you MUST call the 'remember_name' tool."
        )
    return system_prompt

# 3. Define the Agent Logic
def call_model(state: AgentState):
    """The main agent node. It calls the LLM."""
    messages = state["messages"]
    user_name = state.get("user_name")
    
    system_prompt = get_system_prompt(name=user_name)
    
    all_messages = [SystemMessage(content=system_prompt)] + messages
    
    if FLAGS.debug:
        print(f"DEBUG LLM Input Messages:\n{all_messages}\n")

    response = agent_runnable.invoke(all_messages)
    if FLAGS.debug:
        print(f"DEBUG LLM Response Messages:\n{response}\n")

    return {"messages": [response]}

# 4. Define the State Updater Node
def update_state_from_tool_output(state: AgentState):
    """After a tool call, update the state with the tool's output."""
    last_message = state["messages"][-1]
    
    if isinstance(last_message, ToolMessage) and last_message.name == "remember_name":
        try:
            tool_output = json.loads(last_message.content)
            saved_name = tool_output.get("name_saved")
            if saved_name:
                return {"user_name": saved_name}
        except json.JSONDecodeError:
            pass
            
    return {}

# 5. Build the Graph
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_executor)
builder.add_node("update_state", update_state_from_tool_output)

builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: END}
)
builder.add_edge("tools", "update_state")
builder.add_edge("update_state", "agent")

# 6. Define the cleanup function
def cleanup_db(db_file: str):
    """Cleans up the database file and its journal files."""
    if os.path.exists(db_file):
        os.remove(db_file)
    if os.path.exists(f"{db_file}-shm"):
        os.remove(f"{db_file}-shm")
    if os.path.exists(f"{db_file}-wal"):
        os.remove(f"{db_file}-wal")

def print_agent_response(response: dict, run_label: str):
    """Prints the agent's response, handling different content formats."""
    prefix = f"{run_label} Agent Response:"
    agent_response = response['messages'][-1]
    if isinstance(agent_response.content, list) and agent_response.content:
        print(f"{prefix} {agent_response.content[0]['text']}")
    else:
        print(f"{prefix} {agent_response.content}")
    if FLAGS.debug:
        print(f"DEBUG {prefix} (Full):\n{agent_response}\n")

# 7. Compile and Run
def main(_):
    db_file = "langgraph_agent.db"
    try:
        with SqliteSaver.from_conn_string(db_file) as memory:
            app = builder.compile(checkpointer=memory)
            
            session_id_1 = "user_session_1"
            print("--- SESSION 1 (User shares name) ---")
            response = app.invoke(
                {"messages": [HumanMessage(content="Hi, my name is John.")]},
                config={"configurable": {"thread_id": session_id_1}}
            )
            print_agent_response(response, "Run 1")

            print("\n--- SESSION 1 (User returns) ---")
            response = app.invoke(
                {"messages": [HumanMessage(content="What is my name?")]},
                config={"configurable": {"thread_id": session_id_1}}
            )
            print_agent_response(response, "Run 2")

            session_id_2 = "user_session_2"
            print("\n--- SESSION 3 (New user) ---")
            response = app.invoke(
                {"messages": [HumanMessage(content="What is my name?")]},
                config={"configurable": {"thread_id": session_id_2}}
            )
            print_agent_response(response, "Run 3")
    finally:
        cleanup_db(db_file)

if __name__ == "__main__":
    app.run(main)