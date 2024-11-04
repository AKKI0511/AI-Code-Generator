# File: services/task_service.py
from typing import Dict, Optional
from constants import TaskType

class TaskService:
    @staticmethod
    def get_helper_text(task_type: TaskType) -> str:
        helpers = {
            TaskType.CODE_GENERATION: """
            🎯 <b>Best Practices:</b><br>
            • Describe the desired functionality clearly<br>
            • Specify any required libraries<br>
            • Mention performance requirements<br>
            • Include example inputs/outputs<br>
            • State error handling needs
            """,
            TaskType.CODE_REVIEW: """
            🔍 <b>Review Guidelines:</b><br>
            • Paste the complete code<br>
            • Highlight specific concerns<br>
            • Mention performance requirements<br>
            • Request specific aspects to review<br>
            • Include context and constraints
            """,
            TaskType.DOCUMENTATION: """
            📝 <b>Documentation Tips:</b><br>
            • Provide complete code to document<br>
            • Specify documentation style<br>
            • Mention target audience<br>
            • Request specific examples<br>
            • Include usage scenarios
            """,
            TaskType.TESTING: """
            🧪 <b>Testing Requirements:</b><br>
            • Include the code to test<br>
            • Specify test framework preference<br>
            • Mention required test coverage<br>
            • List critical test cases<br>
            • Describe edge cases
            """,
            TaskType.QUERY: """
            💡 <b>Query Guidelines:</b><br>
            • Ask specific questions<br>
            • Provide context<br>
            • Mention related technologies<br>
            • Request examples if needed<br>
            • Specify depth of explanation
            """
        }
        return helpers.get(task_type, "")

    @staticmethod
    def get_placeholder(task_type: TaskType) -> str:
        placeholders = {
            TaskType.CODE_GENERATION: "Example: Create a Python function that processes CSV files, handles errors, and includes type hints...",
            TaskType.CODE_REVIEW: "Paste your code here for review. Include any specific areas of concern...",
            TaskType.DOCUMENTATION: "Paste the code you need documentation for. Specify any particular documentation style...",
            TaskType.TESTING: "Paste the code you need tests for. Mention any specific test cases or scenarios...",
            TaskType.QUERY: "Ask your question about code, best practices, or implementation details..."
        }
        return placeholders.get(task_type, "Describe your task here...")

    @staticmethod
    def get_instructions(task_type: TaskType) -> str:
        instructions = {
            TaskType.CODE_GENERATION: """
            Request code generation with:
            - Detailed requirements
            - Expected functionality
            - Any specific constraints
            """,
            TaskType.CODE_REVIEW: """
            Submit code for review to get:
            - Code quality assessment
            - Bug detection
            - Performance analysis
            - Security review
            """,
            TaskType.DOCUMENTATION: """
            Request documentation for:
            - Function/class documentation
            - Usage examples
            - Parameter descriptions
            """,
            TaskType.TESTING: """
            Generate tests with:
            - Test cases
            - Edge cases
            - Error scenarios
            """,
            TaskType.QUERY: """
            Query the knowledge base for:
            - Code explanations
            - Best practices
            - Implementation guidance
            """
        }
        return instructions.get(task_type, "")
