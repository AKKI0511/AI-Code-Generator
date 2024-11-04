# File: main.py
import streamlit as st
from app import CodeAssistantApp
from models.code_assistant import CodeAssistant

def main():
    # Initialize the CodeAssistant if not already in session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = CodeAssistant()

    # Create and run the application
    app = CodeAssistantApp()
    app.run()

if __name__ == "__main__":
    main()