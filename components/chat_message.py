# File: components/chat_message.py
import streamlit as st
import time
from typing import Optional, Dict, Any

class ChatMessageComponent:
    @staticmethod
    def display(role: str, content: str, result: Optional[Dict[str, Any]] = None):
        message_class = "user-message" if role == "user" else "assistant-message"
        
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div class="message-header">
                    <strong>{'You' if role == "user" else '🤖 AI Assistant'}</strong>
                    <span class="message-time">{time.strftime('%H:%M')}</span>
                </div>
                <div class="message-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if result and result.get("status") == "success":
                ChatMessageComponent._display_result(result)

    @staticmethod
    def _display_result(result: Dict[str, Any]):
        if "code" in result:
            st.code(result["code"], language="python")
            if "filename" in result:
                st.success(f"Saved as: {result['filename']}")
        elif "review" in result:
            st.markdown(result["review"])
        elif "documentation" in result:
            st.markdown(result["documentation"])
        elif "test_code" in result:
            st.code(result["test_code"], language="python")
        elif "response" in result:
            st.markdown(result["response"])
            if "sources" in result:
                with st.expander("View Sources"):
                    for source in result["sources"]:
                        st.markdown(f"- {source[:200]}...")
