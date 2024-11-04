# File: constants.py
from enum import Enum
from pathlib import Path

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    QUERY = "query"

class UIConstants:
    PAGE_TITLE = "AI Code Assistant"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    CHAT_BOTTOM_PADDING = 200
    INPUT_HEIGHT = 100
    
    class CSS:
        DARK_THEME = """
        <style>
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            .stButton button {
                background-color: #262730;
                color: #FAFAFA;
                border: 1px solid #4B4B4B;
            }
            .stTextInput input {
                background-color: #262730;
                color: #FAFAFA;
            }
            .stTextArea textarea {
                background-color: #262730;
                color: #FAFAFA;
            }
            .stSelectbox select {
                background-color: #262730;
                color: #FAFAFA;
            }
            .css-145kmo2 {
                border: 1px solid #4B4B4B;
            }
            .chat-message {
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
                flex-direction: column;
                animation: slideIn 0.3s ease-out;
                transition: all 0.3s ease;
            }
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            .user-message { background-color: #262730; }
            .assistant-message { background-color: #1E1E1E; }
            .code-block {
                background-color: #1E1E1E;
                padding: 1rem;
                border-radius: 0.3rem;
                margin: 1rem 0;
            }
            .sticky-input {
                position: fixed;
                bottom: 0;
                left: 18rem;
                right: 1rem;
                background-color: #0E1117;
                padding: 1rem;
                z-index: 100;
                border-top: 1px solid #4B4B4B;
            }
            .main-content { margin-bottom: 180px; }
            .task-instructions {
                background-color: #1E1E1E;
                border-left: 4px solid #4B4B4B;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 0 0.5rem 0.5rem 0;
            }
            .task-header {
                color: #9EA1FF;
                font-size: 1.1rem;
                margin-bottom: 0.5rem;
            }
            .tooltip {
                position: relative;
                display: inline-block;
                border-bottom: 1px dotted #4B4B4B;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                background-color: #262730;
                color: #FAFAFA;
                text-align: center;
                padding: 0.5rem;
                border-radius: 0.3rem;
                position: absolute;
                z-index: 1;
                width: 300%;
                top: 125%;
                left: 150%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s;
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            .sticky-container {
                position: fixed;
                bottom: 0;
                left: 15.5rem;
                right: 0;
                background-color: #0E1117;
                border-top: 1px solid #4B4B4B;
                padding: 1rem 2rem;
                z-index: 100;
            }
            .input-helper {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                background-color: #1E1E1E;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                max-height: 100px;
                overflow-y: auto;
            }
            .helper-item {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem;
                background-color: #262730;
                border-radius: 0.3rem;
            }
            .helper-icon { font-size: 1.2rem; }
            .helper-text {
                font-size: 0.9rem;
                color: #E0E0E0;
            }
            .input-area {
                display: flex;
                gap: 1rem;
                align-items: flex-start;
            }
            .input-area .textbox { flex-grow: 1; }
            .input-area .submit-btn { flex-shrink: 0; }
        </style>
        """
