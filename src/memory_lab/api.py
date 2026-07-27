import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from memory_lab.agents.graph import build_memory_graph

app = FastAPI(
    title="Memory Lab API",
    description="LLMOps laboratory for testing context retention and conversational memory.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: list[dict[str, str]] = []


# Initialize the graph globally so the MemorySaver instance
# persists across API requests for active sessions.
memory_graph = build_memory_graph()


@app.get("/health")
def health_check():
    """Confirms the laboratory API is up and running."""
    return {"status": "healthy", "service": "memory-lab"}


@app.post("/chat", summary="Send a message and test memory retention")
def chat_with_memory(request: ChatRequest) -> dict[str, Any]:
    """Processes a chat message using the real LangGraph memory agent."""
    try:
        # Map the frontend session_id to LangGraph's thread_id for short-term memory
        config = {"configurable": {"thread_id": request.session_id}}

        # We use a hardcoded user_id for the lab, consistent with the UI
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "user_id": "api_user_01",
        }

        # Invoke the graph. The checkpointer automatically injects previous history.
        result = memory_graph.invoke(initial_state, config=config)

        # Extract the final AI response
        ai_msg = result["messages"][-1].content

        return {
            "status": "success",
            "response": ai_msg,
            "session_id": request.session_id,
        }
    # Any failure in the graph/LLM call should be a clean 500, not an
    # unhandled exception crashing the request.
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Chat execution failed: {e!s}")
