from dotenv import load_dotenv
import os
# Load environment variables (try multiple paths to be robust)
env_paths = ['.env', './notebooks/.env', '../notebooks/.env', 'notebooks/.env']
env_loaded = False
for path in env_paths:
    if load_dotenv(path):
        env_loaded = True
        break

import streamlit as st
from langchain_openai import ChatOpenAI

# 🟢 Must be the first Streamlit command
st.set_page_config(page_title="Advanced AI Chatbot", page_icon="🤖", layout="wide")

# 🎨 Custom CSS for better styling
st.markdown("""
<style>
.chat-container {
    max-height: 600px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 10px;
    margin-bottom: 20px;
}
.stChatMessage {
    margin-bottom: 10px;
}
.sidebar-header {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state FIRST
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🔧 Sidebar with controls
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)

    # Model selection - Only include models that work on OpenRouter
    model_option = st.selectbox(
        "Choose AI Model:",
        ["openai/gpt-4o", "openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"],
        index=0,
        help="Select an AI model. Note: Not all models may be available on OpenRouter."
    )

    # Temperature control
    temperature = st.slider("Creativity (Temperature):", 0.0, 1.0, 0.7, 0.1)

    # Max tokens
    max_tokens = st.slider("Max Response Length:", 100, 250, 500, 950)

    # Clear chat button
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st._rerun()

    # Export chat - Fixed: Use session state to control download
    if st.button("📥 Export Chat"):
        if st.session_state.messages:
            st.session_state.show_download = True
            st._rerun()
        else:
            st.warning("No chat history to export!")

    # Show download button when triggered
    if st.session_state.get("show_download", False):
        chat_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
        st.download_button(
            label="📄 Click to Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            key="download_chat"
        )
        if st.button("❌ Cancel Download"):
            st.session_state.show_download = False
            st.rerun()

# Initialize LLM with user settings (AFTER sidebar controls are defined)
try:
    # Get the API key (Try GPT key first, then Sonnet key as backup)
    api_key = os.getenv("OPENROUTER_GPT_API_KEY") or os.getenv("OPENROUTER_SONET_API_KEY")
    
    if not api_key:
        st.error("❌ API Key not found! Please check your .env file.")
        st.stop()

    llm = ChatOpenAI(
        model=model_option,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=max_tokens,
        temperature=temperature
    )
    st.sidebar.success(f"✅ Model: {model_option}")
except Exception as e:
    st.sidebar.error(f"❌ Model initialization failed: {str(e)}")
    st.stop()

# 🤖 Strong & Attractive Title
st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(to right, #FF4B2B, #FF416C); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🤖 AskBuddy: The Advanced AI Assistant
    </h1>
    <h3 style='text-align: center; color: #555;'>Your intelligent companion for Q&A and creativity.</h3>
    <hr>
""", unsafe_allow_html=True)

# 💬 Chat container with scrolling
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    role = message['role']
    content = message['content']
    with st.chat_message(role):
        st.markdown(content)
st.markdown('</div>', unsafe_allow_html=True)

# 💭 Chat input with placeholder
query = st.chat_input("Ask me anything...", key="chat_input")

if query:
    # Add user message
    st.session_state.messages.append({"role":"user", "content":query})
    with st.chat_message("user"):
        st.markdown(query)

    # Show typing indicator
    with st.chat_message("ai"):
        with st.spinner("🤔 Thinking..."):
            try:
                res = llm.invoke(st.session_state.messages)
                response_content = res.content
            except Exception as e:
                response_content = f"❌ Error: {str(e)}"

        st.markdown(response_content)

    # Save AI response
    st.session_state.messages.append({"role":"ai", "content":response_content})

