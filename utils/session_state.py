# File: utils/session_state.py
from dataclasses import dataclass
from typing import List, Dict, Any
import streamlit as st
from constants import TaskType

@dataclass
class ChatMessage:
    role: str
    content: str
    result: Dict[str, Any] = None

class SessionState:
    @staticmethod
    def initialize():
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'assistant' not in st.session_state:
            st.session_state.assistant = None  # Initialize with your CodeAssistant
        if 'current_task' not in st.session_state:
            st.session_state.current_task = None
        if 'file_upload_key' not in st.session_state:
            st.session_state.file_upload_key = 0
        if "user_input_processed" not in st.session_state:
            st.session_state.user_input_processed = False
        if "rerun_trigger" not in st.session_state:
            st.session_state.rerun_trigger = False

    @staticmethod
    def clear_chat_history():
        st.session_state.chat_history = []
        st.session_state.rerun_trigger = not st.session_state.rerun_trigger
