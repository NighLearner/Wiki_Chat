# app.py

import streamlit as st
import time
from functions import ingest_wikipedia_content, ask
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="Wikipedia RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for dark theme and chat styling
st.markdown("""
<style>
    /* Dark theme colors */
    :root {
        --background-color: #0E1117;
        --text-color: #FAFAFA;
        --chat-user-bg: #2E4B73;
        --chat-assistant-bg: #3C3C3C;
        --border-color: #303030;
    }

    /* Chat container styling */
    .chat-container {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: var(--text-color);
    }

    .user-message {
        background-color: var(--chat-user-bg);
        margin-left: 20%;
        border-radius: 15px 15px 0 15px;
    }

    .assistant-message {
        background-color: var(--chat-assistant-bg);
        margin-right: 20%;
        border-radius: 15px 15px 15px 0;
    }

    /* Topic input styling */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        border: 1px solid var(--border-color);
        border-radius: 5px;
    }

    /* Button styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #45a049;
    }

    /* Divider styling */
    .divider {
        border-top: 1px solid var(--border-color);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Main layout
st.title("Wikipedia RAG Chat 🤖")

# Sidebar for topic selection
with st.sidebar:
    st.header("Topic Selection")
    new_topic = st.text_input("Enter a Wikipedia topic:", 
                             placeholder="e.g., Artificial Intelligence",
                             key="topic_input")
    
    if st.button("Load Topic"):
        if new_topic:
            with st.spinner(f"Loading information about '{new_topic}'..."):
                if ingest_wikipedia_content(new_topic):
                    st.session_state.current_topic = new_topic
                    st.session_state.chat_history = []
                    st.session_state.messages = []
                    st.success(f"Successfully loaded information about '{new_topic}'!")
                else:
                    st.error("Failed to load topic. Please try another one.")
        else:
            st.warning("Please enter a topic.")
    
    # Show current topic
    if st.session_state.current_topic:
        st.markdown("---")
        st.markdown(f"**Current Topic:** {st.session_state.current_topic}")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

# Main chat area
if st.session_state.current_topic:
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-container user-message">
                        {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-container assistant-message">
                        {message["content"]}
                    </div>
                """, unsafe_allow_html=True)

    # Input area
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    with st.container():
        query = st.text_input("Ask a question:", key="query_input", 
                             placeholder=f"Ask something about {st.session_state.current_topic}...")
        
        if st.button("Send"):
            if query:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": query})
                
                # Get response
                with st.spinner("Thinking..."):
                    result = ask(query, st.session_state.chat_history)
                    
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                
                # Update chat history if response was meaningful
                if result["context"] and not result["answer"].startswith(
                    ("I don't have enough information", "No relevant information")
                ):
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
                
                # Clear input and rerun to update chat
                st.rerun()
else:
    st.info("👈 Please select a topic from the sidebar to start chatting!")