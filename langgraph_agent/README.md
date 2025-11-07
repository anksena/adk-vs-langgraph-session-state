# LangGraph Agent

This agent is designed to demonstrate a conversational AI that can remember a user's name across different sessions. It uses LangGraph to manage the agent's state and tools to interact with an external database.

## Setup

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
    *   Copy the `.env.example` file to a new file named `.env`.
    *   Add your `GOOGLE_API_KEY` to the `.env` file.

## Usage

To run the agent, execute the following command from within the `langgraph_agent` directory:

```bash
python graph.py