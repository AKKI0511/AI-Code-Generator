from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from enum import Enum
import json
from dataclasses import dataclass
from llama_index.core import (
    VectorStoreIndex,
    Document,
    ServiceContext,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate

# from llama_index.core.query_engine import QueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.response_synthesizers import ResponseMode
import dotenv
import os

# Import helper classes from local file
import path
import sys

# directory reach
directory = path.Path(__file__).absolute()

# setting path
sys.path.append(directory.parent.parent)

# importing
from constants import TaskType

# Load environment variables

dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# class TaskType(Enum):
#     CODE_GENERATION = "code_generation"
#     CODE_REVIEW = "code_review"
#     DOCUMENTATION = "documentation"
#     TESTING = "testing"
#     QUERY = "query"

@dataclass
class TaskConfig:
    """Configuration for different task types"""

    prompt_template: str
    model_name: str
    temperature: float
    response_mode: Optional[ResponseMode] = None


class CodeAssistant:
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "output",
        persist_dir: str = "storage",
        openai_api_key: Optional[str] = None,
    ):
        """Initialize the Enhanced Code Assistant"""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.persist_dir = Path(persist_dir)
        self.openai_api_key = openai_api_key
        self.groq_api_key = GROQ_API_KEY
        self.llama_cloud_api_key = LLAMA_CLOUD_API_KEY

        # Create necessary directories
        for directory in [self.data_dir, self.output_dir, self.persist_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize components and task configurations
        self.setup_components()
        self.setup_task_configs()
        self.setup_document_index()

        # Define code generation prompt
        # self.code_prompt = PromptTemplate(
        #     """You are an expert code generator. Given the following request, generate clean,
        #     well-documented Python code that solves the problem. Include proper error handling,
        #     type hints, and docstrings.

        #     Request: {prompt}

        #     Generate code that:
        #     1. Is production-ready and follows PEP standards
        #     2. Includes comprehensive error handling
        #     3. Has clear documentation
        #     4. Uses type hints
        #     5. Is efficient and maintainable

        #     Code:
        #     """
        # )

    def setup_components(self):
        """Initialize various LLM models and embedding model"""
        try:
            # Initialize different LLMs for different tasks
            self.llms = {
                "mistral": Ollama(model="mistral:latest", request_timeout=3600),
                "codellama": Ollama(model="codellama", request_timeout=3600),
                "llama": Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
            }

            # Add OpenAI if API key is provided
            if self.openai_api_key:
                self.llms["gpt4"] = OpenAI(
                    model="gpt-4-turbo-preview",
                    api_key=self.openai_api_key,
                    temperature=0.7,
                )

            # Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

            # Configure service context
            Settings.embed_model = self.embed_model
            Settings.node_parser = SentenceSplitter()

            logger.info("Successfully initialized all components")
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def setup_task_configs(self):
        """Setup configurations for different tasks"""
        self.task_configs = {
            TaskType.CODE_GENERATION: TaskConfig(
                prompt_template="""You are an expert code generator. Generate production-ready Python code for the following request:
                
                Request: {prompt}
                
                Requirements:
                1. Include error handling
                2. Add type hints
                3. Write clear docstrings
                4. Follow PEP standards
                5. Make code efficient and maintainable
                
                Code:""",
                model_name="llama",
                temperature=0.7,
            ),
            TaskType.CODE_REVIEW: TaskConfig(
                prompt_template="""Review the following code and provide detailed feedback:
                
                Code: {prompt}
                
                Analyze:
                1. Code quality and style
                2. Potential bugs
                3. Performance issues
                4. Security concerns
                5. Suggested improvements
                
                Review:""",
                model_name="gpt4" if self.openai_api_key else "mistral",
                temperature=0.3,
            ),
            TaskType.DOCUMENTATION: TaskConfig(
                prompt_template="""Generate comprehensive documentation for the following code:
                
                Code: {prompt}
                
                Include:
                1. Overview
                2. Usage examples
                3. Function/class descriptions
                4. Parameter details
                5. Return value descriptions
                
                Documentation:""",
                model_name="mistral",
                temperature=0.4,
            ),
            TaskType.TESTING: TaskConfig(
                prompt_template="""Generate unit tests for the following code:
                
                Code: {prompt}
                
                Requirements:
                1. Use pytest framework
                2. Include edge cases
                3. Add proper assertions
                4. Test error cases
                5. Add test documentation
                
                Tests:""",
                model_name="llama",
                temperature=0.5,
            ),
            TaskType.QUERY: TaskConfig(
                prompt_template="{prompt}",
                model_name="mistral",
                temperature=0.3,
                response_mode=ResponseMode.TREE_SUMMARIZE,
            ),
        }

    def setup_document_index(self):
        """Setup or load document index"""
        try:
            # Try to load existing index
            if (self.persist_dir / "docstore.json").exists():
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.persist_dir)
                )
                self.index = load_index_from_storage(storage_context)
                logger.info("Loaded existing document index")
            else:
                # Create new index if none exists
                self._create_new_index()

            # Setup query engine
            self.query_engine = self.index.as_query_engine(
                llm=self.llms["mistral"],
                response_mode=ResponseMode.TREE_SUMMARIZE,
            )

        except Exception as e:
            logger.error(f"Document index setup error: {str(e)}")
            raise

    def _create_new_index(self):
        """Create new document index"""
        if list(self.data_dir.glob("*")):  # Check if data directory has files
            documents = SimpleDirectoryReader(str(self.data_dir)).load_data()

            self.index = VectorStoreIndex.from_documents(
                documents, embed_model=self.embed_model
            )

            # Persist index
            self.index.storage_context.persist(persist_dir=str(self.persist_dir))
            logger.info("Created and persisted new document index")
        else:
            # Create empty index if no documents
            self.index = VectorStoreIndex([])
            logger.info("Created empty document index")

    def process_task(self, task_type: TaskType, prompt: str) -> Dict[str, Any]:
        """Process different types of tasks"""
        try:
            config = self.task_configs[task_type]

            if task_type == TaskType.QUERY:
                return self._handle_query(prompt)

            # Format prompt based on task type
            formatted_prompt = PromptTemplate(config.prompt_template).format(
                prompt=prompt
            )

            # Get appropriate LLM
            llm = self.llms[config.model_name]

            # Generate response
            response = llm.complete(formatted_prompt)

            # Process response based on task type
            if task_type == TaskType.CODE_GENERATION:
                return self._handle_code_generation(prompt, response.text)
            elif task_type == TaskType.CODE_REVIEW:
                return self._handle_code_review(response.text)
            elif task_type == TaskType.DOCUMENTATION:
                return self._handle_documentation(response.text)
            elif task_type == TaskType.TESTING:
                return self._handle_testing(prompt, response.text)

        except Exception as e:
            logger.error(f"Task processing error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _handle_query(self, query: str) -> Dict[str, Any]:
        """Handle document queries"""
        try:
            response = self.query_engine.query(query)
            return {
                "status": "success",
                "response": str(response),
                "sources": [node.node.text for node in response.source_nodes],
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _handle_code_generation(self, prompt: str, response: str) -> Dict[str, Any]:
        """Handle code generation results"""
        code_block = self._extract_code(response)
        filename = self._generate_filename(prompt) + ".py"
        self._save_code(filename, code_block)

        return {"status": "success", "code": code_block, "filename": filename}

    # TODO
    def _handle_code_review(self, response: str) -> Dict[str, Any]:
        """Handle code review results"""
        return {"status": "success", "review": response}

    # TODO
    def _handle_documentation(self, response: str) -> Dict[str, Any]:
        """Handle documentation generation results"""
        doc_filename = self._generate_filename("documentation")
        self._save_code(f"{doc_filename}.md", response)
        return {"status": "success", "documentation": response}

    def _handle_testing(self, original_code: str, test_code: str) -> Dict[str, Any]:
        """Handle test generation results"""
        test_filename = f"test_{self._generate_filename(original_code)}.py"
        self._save_code(test_filename, self._extract_code(test_code))

        return {
            "status": "success",
            "test_code": self._extract_code(test_code),
            "test_filename": test_filename,
        }

        # def generate_code(self, prompt: str) -> Dict[str, Any]:
        # """Generate code based on the prompt"""
        # try:
        #     # Format the prompt
        #     formatted_prompt = self.code_prompt.format(prompt=prompt)

        #     # Generate response using LLM
        #     response = self.llm.complete(formatted_prompt)

        #     # Extract code from response
        #     code_block = self._extract_code(response.text)

        #     # Generate filename
        #     filename = self._generate_filename(prompt)

        #     # Save the code
        #     self._save_code(filename, code_block)

        #     return {"status": "success", "code": code_block, "filename": filename}
        # except Exception as e:
        #     logger.error(f"Code generation error: {str(e)}")
        #     return {"status": "error", "error": str(e)}

    def _extract_code(self, response: str) -> str:
        """Extract code block from the response"""
        # Simple extraction - can be enhanced based on needs
        lines = response.split("\n")
        code_lines = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```python"):
                in_code_block = True
            elif line.strip().startswith("```") and in_code_block:
                in_code_block = False
            elif in_code_block:
                code_lines.append(line)
            elif not any(line.strip().startswith("```") for line in lines):
                # If no code blocks found, treat entire response as code
                return response

        return "\n".join(code_lines) if code_lines else response

    def _generate_filename(self, prompt: str) -> str:
        """Generate a valid filename from the prompt"""
        # Convert prompt to snake case and clean it up
        filename = "_".join(
            "".join(c if c.isalnum() else " " for c in prompt.lower()).split()
        )
        return f"{filename[:min(30, len(filename))]}"  # Truncate to reasonable length

    def _save_code(self, filename: str, code: str) -> bool:
        """Save generated code to file"""
        try:
            file_path = self.output_dir / filename
            file_path.write_text(code)
            logger.info(f"Successfully saved code to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return False

    def refresh_document_index(self):
        """Refresh the document index with any new files"""
        self._create_new_index()
        logger.info("Document index refreshed")


# def main():
#     """Main execution function"""
#     try:
#         # Initialize the assistant
#         assistant = CodeAssistant()
#         logger.info("Code Assistant initialized successfully")

#         while True:
#             print("\nAvailable tasks:")
#             for task in TaskType:
#                 print(f"{task.value}: {task.name}")

#             task_input = input("\nEnter task type (or 'q' to quit): ").strip().lower()

#             if task_input == "q":
#                 break

#             try:
#                 task_type = TaskType(task_input)
#             except ValueError:
#                 print("Invalid task type. Please try again.")
#                 continue

#             prompt = input("\nEnter your prompt: ").strip()

#             if not prompt:
#                 print("Please enter a valid prompt")
#                 continue

#             result = assistant.process_task(task_type, prompt)

#             if result["status"] == "success":
#                 if task_type == TaskType.CODE_GENERATION:
#                     print("\nGenerated Code:")
#                     print(result["code"])
#                     print(f"\nSaved to file: {result['filename']}")
#                 elif task_type == TaskType.QUERY:
#                     print("\nResponse:")
#                     print(result["response"])
#                     print("\nSources:")
#                     for source in result["sources"]:
#                         print(f"\n- {source[:200]}...")
#                 else:
#                     print("\nResult:")
#                     print(json.dumps(result, indent=2))
#             else:
#                 print(f"Error: {result.get('error', 'Unknown error occurred')}")

#     except KeyboardInterrupt:
#         print("\nExiting...")
#     except Exception as e:
#         logger.error(f"Application error: {str(e)}")


# if __name__ == "__main__":
#     main()
