import operator
import os
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from memory_lab.memory.chroma_store import LongTermMemory


# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str


# Initialize LLM & Memory
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11435")
model_name = os.getenv("OLLAMA_DEFAULT_MODEL", "phi3:latest")

llm = ChatOllama(base_url=ollama_host, model=model_name, temperature=0.3)
ltm = LongTermMemory()


# 2. Main Processing Node
def process_message_node(state: AgentState):
    """
    Retrieves long-term facts, combines them with the short-term conversation history,
    and generates a personalized response.
    """
    user_id = state["user_id"]
    messages = state["messages"]

    # Get the latest message from the user
    last_user_message = messages[-1].content if messages else ""

    # --- MEMORY EXTRACTION (Simplified for Lab) ---
    # In a production app, we would use a Tool or another LLM prompt to extract facts.
    # Here, we use simple keyword heuristics to simulate fact saving.
    lower_msg = last_user_message.lower()
    trigger_words = [
        "i like",
        "i love",
        "my name is",
        "i live in",
        "i am",
        "my favorite",
    ]
    if any(trigger in lower_msg for trigger in trigger_words):
        ltm.save_fact(user_id, last_user_message)

    # --- MEMORY RETRIEVAL ---
    # Fetch relevant facts from ChromaDB
    facts = ltm.retrieve_relevant_facts(user_id, last_user_message)
    facts_str = "\n- ".join(facts) if facts else "No specific facts known yet."

    # --- PROMPT CONSTRUCTION ---
    sys_msg = SystemMessage(
        content=f"You are a helpful, highly personalized AI assistant.\n"
        f"You have a memory of the user. Use these retrieved facts to personalize your response:\n"
        f"- {facts_str}\n\n"
        f"Be conversational and natural. Do not explicitly say 'I see in my database'."
    )

    # Invoke LLM with System Prompt + Chat History (managed automatically by LangGraph)
    response = llm.invoke([sys_msg] + messages)

    return {"messages": [response]}


# 3. Build and Compile the Graph
def build_memory_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("process_message", process_message_node)
    workflow.set_entry_point("process_message")
    workflow.add_edge("process_message", END)

    # Checkpointer provides Short-Term Memory (Session Threading)
    memory_saver = MemorySaver()

    return workflow.compile(checkpointer=memory_saver)
