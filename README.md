# AI Code Assistant

A sophisticated Streamlit-based application that delivers AI-powered code assistance, performance profiling, and security scanning. By leveraging multiple language models and advanced vector search capabilities, the system provides seamless and context-aware support for various code-related tasks.

---

## Features

### Core Capabilities
- **Multi-Model AI Processing**
  - Supports Mistral (Ollama), CodeLlama (Ollama), LLaMA 3 (Groq), and GPT-4 (optional OpenAI integration).

- **Vector-Based Document Management**
  - Persistent index storage and real-time updates.
  - Context-aware querying for efficient document retrieval.

- **Task Processing**
  - Production code generation.
  - Comprehensive code review and analysis.
  - Automated documentation and test case creation.
  - Contextual and intelligent querying.

- **Profiler**
  - Real-time performance monitoring and optimization suggestions.
  - Dependency management and detailed reporting.
  - AI-powered optimization for Python code.
  - Supports both synchronous and asynchronous code execution.

### Security Scanner
- **Advanced Code Security Scanning**
  - Quick, deep, and custom scan options.
  - Extensive vulnerability detection.
  - Git repository scanning with authentication support.
  - Detailed reports with severity-based filtering.
  - Export results in JSON or PDF formats.

### Technical Implementation
- Streamlit-based intuitive user interface.
- Hugging Face embedding model for vector search.
- Persistent vector store indexing for document management.
- Multi-model task routing system for optimized performance.
- Integrated Profiler for performance monitoring and optimization.

---

## Installation

### Prerequisites
- Python 3.8+
- Installed dependencies for Ollama and Groq APIs.
- Optional: OpenAI API access for GPT-4.

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AKKI0511/AI-Code-Generator.git
   cd AI-Code-Generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```env
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key  # Optional
   ```

4. Launch the application:
   ```bash
   streamlit run main.py
   ```

---

## Architecture

### Directory Structure
```
project/
├── main.py                 # Application entry point
├── app.py                  # Core application logic
├── constants.py            # System constants
├── requirements.txt        # Dependencies
├── components/             # UI components
├── models/                 # AI models and core logic
├── services/               # Business logic
└── utils/                  # Utility functions
```

### Component Overview
- **`main.py`**: Initializes and configures the application.
- **`app.py`**: Manages the Streamlit interface and user interaction.
- **`code_assistant.py`**: Core AI processing and task orchestration.
- **`task_service.py`**: Task-specific business logic.
- **`session_state.py`**: Manages application states across sessions.

---

## Functional Specifications

### Code Generation
- **Input**: Natural language description.
- **Output**: Production-ready Python code.
- **Features**:
  - Implements error handling and type hints.
  - Ensures PEP compliance and performance optimization.
  - Auto-generates documentation.

### Code Review
- **Capabilities**:
  - Evaluates code quality and identifies bugs.
  - Analyzes performance and security.
  - Provides optimization recommendations.

### Profiler
- **Capabilities**:
  - Monitors execution time, memory usage, and CPU utilization.
  - Provides AI-powered optimization suggestions.
  - Generates detailed performance reports.
  - Supports both synchronous and asynchronous code execution.

### Documentation Generation
- **Output Components**:
  - System overviews, implementation details, API documentation, and usage examples.
  - Parameter specifications with clear formatting.

### Test Case Creation
- **Features**:
  - Generates test cases using Pytest framework.
  - Covers edge cases, error scenarios, and assertions.
  - Includes test documentation.

### Query Processing
- **Highlights**:
  - Provides context-aware responses with source citation.
  - Aids code comprehension and implementation guidance.
  - Recommends best practices.

---

## Security Scanner

### Scan Options
- **Quick Scan**: Performs basic security checks.
- **Deep Scan**: Executes comprehensive analysis.
- **Custom Scan**: Allows user-defined checks.

### Input Methods
- Upload code files or input code snippets directly.
- Scan Git repositories with authentication.

### Output
- Generates detailed vulnerability reports.
- Supports export options in JSON and PDF formats.
- Tracks historical data for trend analysis.

---

## Configuration

### Environment Variables
```
GROQ_API_KEY          # Required for LLaMA 3 integration
OPENAI_API_KEY        # Optional for GPT-4
LLAMA_CLOUD_API_KEY   # Optional for cloud-based services
```

### Model Configuration
- **Embedding Model**: BAAI/bge-small-en-v1.5.
- **Vector Store**: LlamaIndex for efficient indexing.
- **Node Parser**: Configured with SentenceSplitter for optimal parsing.

---

## Usage Instructions

### Task Selection
1. Choose the desired task from the available options.
2. Input the task-specific details or requirements.
3. Review and refine the generated output.
4. Access saved outputs from the designated directory.

### Document Management
1. Upload documents via the user interface.
2. Allow the system to index content automatically.
3. Query the indexed content using natural language.
4. Refresh indexes as needed to include new data.

---

## Development Guidelines

### Extensibility
- Integrate additional models through a modular interface.
- Add new task types with minimal disruption to existing architecture.
- Enhance UI components for improved usability.

### Best Practices
- Adhere to PEP standards for code quality.
- Implement robust error handling mechanisms.
- Use type hints for better readability and debugging.
- Document all new features and include unit tests.

---

## Contributions
Contributions are welcome! Please submit pull requests or open issues to improve the project.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.
