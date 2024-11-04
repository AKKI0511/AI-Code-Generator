import streamlit as st
from constants import TaskType, UIConstants
from utils.session_state import SessionState
from services.task_service import TaskService
from components.chat_message import ChatMessageComponent
from components.file_uploader import FileUploaderComponent

class CodeAssistantApp:
    def __init__(self):
        self._configure_page()
        SessionState.initialize()

    def _configure_page(self):
        st.set_page_config(
            page_title=UIConstants.PAGE_TITLE,
            page_icon=UIConstants.PAGE_ICON,
            layout=UIConstants.LAYOUT,
            initial_sidebar_state=UIConstants.INITIAL_SIDEBAR_STATE,
        )
        st.markdown(UIConstants.CSS.DARK_THEME, unsafe_allow_html=True)

    def render_sidebar(self):
        with st.sidebar:
            st.title("🤖 AI Code Assistant")
            st.markdown("---")

            st.subheader("Task Selection")
            task_options = {task.value: task.name for task in TaskType}
            st.selectbox(
                "Select Task",
                options=list(task_options.keys()),
                format_func=lambda x: task_options[x],
                key="task_select",
            )

            st.markdown("---")
            st.subheader("File Upload")
            FileUploaderComponent.render(st.session_state.assistant)

            st.markdown("---")
            if st.button("Clear Chat History"):
                SessionState.clear_chat_history()

    def handle_task_execution(self):
        task_type = TaskType(st.session_state.task_select)
        prompt = st.session_state.user_input

        if not prompt:
            st.warning("Please enter a prompt")
            return

        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.spinner("Processing your request..."):
            result = st.session_state.assistant.process_task(task_type, prompt)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": "Here's what I've prepared:",
                "result": result,
            }
        )

        st.session_state.user_input_processed = True
        st.rerun()
        st.session_state.user_input = ""

    def render_main_content(self):
        st.title("💻 Interactive Code Assistant")

        # Create scrollable container for chat history
        st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            ChatMessageComponent.display(
                role=message["role"],
                content=message["content"],
                result=message.get("result"),
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Sticky input container
        st.markdown('<div class="sticky-input-container">', unsafe_allow_html=True)

        task_type = TaskType(st.session_state.task_select)
        helper_text = TaskService.get_helper_text(task_type)
        st.markdown(
            f'<div class="tooltip">ℹ️ Input Helper<span class="tooltiptext">{helper_text}</span></div>',
            unsafe_allow_html=True,
        )

        # Input area with columns for better layout
        col1, col2 = st.columns([100, 1])

        with col1:
            st.text_area(
                "Enter your prompt",
                key="user_input",
                height=UIConstants.INPUT_HEIGHT,
                placeholder=TaskService.get_placeholder(task_type),
            )

        with col1:
            if st.button("Submit", use_container_width=True):
                self.handle_task_execution()

        st.markdown("</div>", unsafe_allow_html=True)

    def run(self):
        """Main entry point for the application"""
        self.render_sidebar()
        self.render_main_content()
