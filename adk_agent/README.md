# ADK Agent (Stateful)

This agent demonstrates a conversational AI that uses the ADK (Agent Development Kit) to manage state in a single session.

## Setup

To setup the agent, execute the following command from within the `adk_agent` directory:

1.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

2.  **Activate the virtual environment:**
    *   On Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create an environment file:**
    *   Copy the `.env.example` file (if present) to `session_persistence_agent/.env`.
    *   Add your `GEMINI_API_KEY` to the `.env` file.

## Usage

To run the agent, execute the following command from within the `adk_agent` directory:

```bash
python -m session_persistence_agent.runner
```