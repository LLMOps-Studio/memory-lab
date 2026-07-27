import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from memory_lab.agents.graph import build_memory_graph
from memory_lab.memory.feedback_db import FeedbackDB

# 1. Page Configurations
st.set_page_config(page_title="Memory Lab", page_icon="🧠", layout="centered")

# Apply central theme (Purple for Memory Lab)
st.markdown("<style>:root {--accent: #8B5CF6;}</style>", unsafe_allow_html=True)


# 2. Initialize Core Services
@st.cache_resource
def get_services():
    return build_memory_graph(), FeedbackDB()


graph, feedback_db = get_services()

# 3. Session State Management
if "session_id" not in st.session_state:
    # A unique thread ID for LangGraph's short-term memory checkpointer
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    # Hardcoded user ID for lab purposes (simulating a logged-in user)
    st.session_state.user_id = "user_alpha_01"

st.title("🧠 Memory Lab")
st.markdown(
    "Chat with the personalized agent. Try stating facts like *'I love playing guitar'* or *'My favorite color is blue'*, then ask it later to test its memory!"
)

# LangGraph configuration requiring the thread_id for short-term memory mapping
config = {"configurable": {"thread_id": st.session_state.session_id}}

# 4. Render Chat History from LangGraph State
state = graph.get_state(config)
messages = state.values.get("messages", []) if state.values else []

for msg in messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# 5. Chat Input Processing
if prompt := st.chat_input("Tell me something about yourself..."):
    # Display the new user message
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking and retrieving memories from ChromaDB..."):
        # Send message to LangGraph
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "user_id": st.session_state.user_id,
        }
        result = graph.invoke(initial_state, config=config)

        # Extract and display the AI response
        ai_msg = result["messages"][-1].content
        st.chat_message("assistant").write(ai_msg)

        # Temporarily store the last interaction for the feedback buttons
        st.session_state.last_user_msg = prompt
        st.session_state.last_ai_msg = ai_msg
        st.rerun()

# 6. Feedback Mechanism
# Only show feedback buttons if an AI response was just generated and hasn't been voted on yet
if "last_ai_msg" in st.session_state:
    st.markdown("---")
    st.caption("Was the memory retrieval accurate for this response?")

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("👍 Accurate"):
            feedback_db.log_feedback(
                st.session_state.session_id,
                st.session_state.last_user_msg,
                st.session_state.last_ai_msg,
                1,
            )
            del st.session_state.last_user_msg
            del st.session_state.last_ai_msg
            st.rerun()
    with col2:
        if st.button("👎 Hallucinated/Forgot"):
            feedback_db.log_feedback(
                st.session_state.session_id,
                st.session_state.last_user_msg,
                st.session_state.last_ai_msg,
                -1,
            )
            del st.session_state.last_user_msg
            del st.session_state.last_ai_msg
            st.rerun()
