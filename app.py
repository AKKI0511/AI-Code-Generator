import streamlit as st
from constants import TaskType, UIConstants
from tools.profiler_page import ProfilerPage
from tools.security_scanner_page import SecurityScannerPage
from utils.session_state import SessionState
from services.task_service import TaskService
from components.chat_message import ChatMessageComponent
from components.file_uploader import FileUploaderComponent

class CodeAssistantApp:
    def __init__(self):
        """Initialize the CodeAssistantApp by configuring the page and setting up navigation."""
        self._configure_page()
        SessionState.initialize()
        self._setup_navigation()

    def _configure_page(self):
        """Configure the Streamlit page settings and apply the dark theme."""
        st.set_page_config(
            page_title=UIConstants.PAGE_TITLE,
            page_icon=UIConstants.PAGE_ICON,
            layout=UIConstants.LAYOUT,
            initial_sidebar_state=UIConstants.INITIAL_SIDEBAR_STATE,
        )
        st.markdown(UIConstants.CSS.DARK_THEME, unsafe_allow_html=True)

    def _setup_navigation(self):
        """Set up the sidebar navigation using a radio button to switch between pages."""
        st.sidebar.radio(
            "# Select Page",
            options=["Code Assistant", "Security Scanner", "Code Profiler"],
            index=0 if st.session_state.active_tab == "Code Assistant" else (1 if st.session_state.active_tab == "Security Scanner" else 2),
            key="active_tab",
            format_func=lambda x: x,
        )

    def render_sidebar(self):
        """Render the sidebar only for the 'Code Assistant' page."""
        if st.session_state.active_tab == "Code Assistant":
            with st.sidebar:
                st.markdown("---")
                self.render_task_selection()

    def render_task_selection(self):
        """Render the task selection and file upload components in the sidebar."""
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
        """Handle the execution of the selected task based on user input."""
        task_type = TaskType(st.session_state.task_select)
        prompt = st.session_state.user_input

        if not prompt:
            st.warning("Please enter a prompt")
            return
        if not task_type:
            st.warning("Please select a task type")
            return

        # Append user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Process the task and append the result to chat history
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

    def render_main_content(self):
        """Render the main content area for the 'Code Assistant' page."""
        st.title("💻 Interactive Code Assistant")

        # Create a scrollable container for chat history
        st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            ChatMessageComponent.display(
                role=message["role"],
                content=message["content"],
                result=message.get("result"),
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Sticky input container for user prompt
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
        """Main entry point for the application. Render sidebar and main content based on the active tab."""
        # Render sidebar conditionally based on the active tab
        self.render_sidebar()

        # Render content based on the active tab
        if st.session_state.active_tab == "Code Assistant":
            self.render_main_content()
        elif st.session_state.active_tab == "Security Scanner":
            SecurityScannerPage().render()
        elif st.session_state.active_tab == "Code Profiler":
            ProfilerPage().render()
