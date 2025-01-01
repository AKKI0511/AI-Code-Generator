# File: utils/session_state.py
from dataclasses import dataclass
from typing import List, Dict, Any
import streamlit as st
from constants import TaskType
from models.vulnerability_scanner.helper import SecurityScanResult

@dataclass
class ChatMessage:
    role: str
    content: str
    result: Dict[str, Any] = None

class SessionState:
    
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
        if "active_tab" not in st.session_state:
            st.session_state.active_tab = "Code Assistant"
        if "scan_history" not in st.session_state:
            st.session_state.scan_history = []
        if "current_scan_id" not in st.session_state:
            st.session_state.current_scan_id = None
        if "filter_severity" not in st.session_state:
            st.session_state.filter_severity = ["High", "Medium", "Low"]
        if "scan_result" not in st.session_state:
            st.session_state.scan_result = None
        if 'profiler_code' not in st.session_state:
            st.session_state.profiler_code = ""
        if 'profiler_results' not in st.session_state:
            st.session_state.profiler_results = None
        if 'input_method' not in st.session_state:
            st.session_state.input_method = "Enter Code"
            
    def clear_chat_history():
        st.session_state.chat_history = []
        st.session_state.rerun_trigger = not st.session_state.rerun_trigger
