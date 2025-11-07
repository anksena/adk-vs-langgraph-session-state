import asyncio
import os
import sys
import logging
from absl import app
from absl import flags
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai.types import Content, Part
import google.generativeai as genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from session_persistence_agent.agent import root_agent

DB_FILE = "agent_session_data.db"
APP_NAME = "stateful_session_app"
FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", False, "Enable debug logging.")

def configure_llm():
    """Configures the LLM."""
    logging.getLogger().setLevel(logging.WARNING)
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)

def setup_database_session_service():
    """Sets up a session service that uses a database for storage."""
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    db_url = f"sqlite:///{DB_FILE}"
    return DatabaseSessionService(db_url=db_url)

async def create_session(
    session_service: DatabaseSessionService,
    app_name: str,
    user_id: str,
    session_id: str,
    run_label: str,
):
    """Creates a session, gets it, and prints the state."""
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if FLAGS.debug:
        session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        print(f"{run_label} Initial state for session {session_id}: {session.state}")

async def run_session(
    runner: Runner,
    user_id: str,
    session_id: str,
    message: str,
    run_label: str,
):
    """Runs a session and prints the final response."""
    user_message = Content(parts=[Part(text=message)])
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        if FLAGS.debug:
            print(f"{run_label} Event: {event}")
        if event.is_final_response():
            print(f"{run_label} Final response: {event.content.parts[0].text}")
            break

async def create_and_run_session(
    runner: Runner,
    session_service: DatabaseSessionService,
    app_name: str,
    user_id: str,
    session_id: str,
    message: str,
    run_label: str,
):
    """Creates a session, gets it, and prints the state."""
    await create_session(
        session_service, app_name, user_id, session_id, run_label
    )
    await run_session(
        runner,
        user_id,
        session_id,
        message,
        run_label,
    )

async def main(argv):
    """Main function to configure and run the agent."""
    configure_llm()

    persistent_session_service = setup_database_session_service()

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=persistent_session_service,
    )

    print("\n[Run 1: User shares name in NEW session]")
    USER_ID, USER_NAME, SESSION_ID_1, RUN_ID_1 = "user_John", "John", "session_789", "Run 1"
    await create_and_run_session(
        runner,
        persistent_session_service,
        APP_NAME,
        USER_ID,
        SESSION_ID_1,
        f"My name is {USER_NAME}.",
        RUN_ID_1,
    )

    # --- Run 2: Ask for the name (SAME session) ---
    print("\n[Run 2: User asks for name in SAME session]")
    RUN_ID_2 = "Run 2"
    await run_session(
        runner,
        USER_ID,
        SESSION_ID_1, # Note: SAME session_id
        "What is my name?",
        RUN_ID_2,
    )    

    # --- Run 3: Ask for the name (NEW session) ---
    print("\n[Run 3: User asks for name in NEW session]")
    SESSION_ID_2, RUN_ID_3 = "session_101", "Run 3"
    await create_and_run_session(
        runner,
        persistent_session_service,
        APP_NAME,
        USER_ID,
        SESSION_ID_2, # Note: NEW session_id
        "What is my name?",
        RUN_ID_3,
    )


    persistent_session_service.db_engine.dispose()

def main_wrapper(argv):
    """Wrapper for the main function."""
    try:
        asyncio.run(main(argv))
    finally:
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)

if __name__ == "__main__":
    app.run(main_wrapper)
