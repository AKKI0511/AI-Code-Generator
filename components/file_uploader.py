# File: components/file_uploader.py
import streamlit as st
from pathlib import Path
import os

class FileUploaderComponent:
    @staticmethod
    def render(assistant):
        # Create data directory if it doesn't exist
        os.makedirs(assistant.data_dir, exist_ok=True)

        uploaded_file = st.file_uploader(
            "Upload Python files for analysis",
            type=["py", "txt", "md", "pdf"],
            key=f"file_uploader_{st.session_state.file_upload_key}"
        )
        
        if uploaded_file:
            try:
                # Ensure absolute path is used
                abs_data_dir = os.path.abspath(assistant.data_dir)
                file_path = Path(abs_data_dir) / uploaded_file.name

                # Save file content
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                # Verify file was saved
                if not file_path.exists():
                    raise Exception("File was not saved successfully")

                assistant.refresh_document_index()
                st.success(f"Successfully uploaded {uploaded_file.name} to {file_path}")
                st.session_state.file_upload_key += 1
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
