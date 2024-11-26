# AI Code Assistant

A sophisticated Streamlit-based application offering AI-powered code assistance and security scanning through multiple language models and advanced vector search capabilities. The system leverages various LLMs for different code-related tasks while maintaining context through vector-based document indexing.

## Features

### Core Capabilities
- **Multi-Model AI Processing**
  - Mistral (Ollama)
  - CodeLlama (Ollama)
  - LLaMA 3 (Groq)
  - GPT-4 (Optional OpenAI integration)

- **Vector-Based Document Management**
  - Persistent index storage
  - Real-time index updates
  - Context-aware querying
  - Efficient document retrieval

- **Task Processing**
  - Production code generation
  - Code review and analysis
  - Documentation generation
  - Test case creation
  - Contextual querying

### Security Scanner
- **Advanced Code Security Scanning**
  - Quick, Deep, and Custom scan options
  - Comprehensive vulnerability detection
  - Git repository scanning with authentication support
  - Detailed vulnerability reports with severity filters
  - Export scan results in JSON or PDF formats

### Technical Implementation
- Streamlit-based user interface
- Hugging Face embedding model integration
- Vector store indexing for document management
- Multi-model task routing system
- Persistent storage management

## Installation

### System Requirements
- Python 3.8+
- Ollama installation
- Groq API access
- Optional: OpenAI API access

### Setup Process
1. Repository Clone:
```bash
git clone https://github.com/AKKI0511/AI-Code-Generator.git
cd AI-Code-Generator
```

2. Dependency Installation:
```bash
pip install -r requirements.txt
```

3. Environment Configuration:
```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional
```

### Execution
Launch the application:
```bash
streamlit run main.py
```

## Architecture

### Directory Structure
```
project/
├── main.py                 # Application entry point
├── app.py                  # Core application logic
├── constants.py            # System constants
├── requirements.txt        # Dependencies
├── components/            # UI components
├── models/               # AI models and core logic
├── services/            # Business logic
└── utils/              # Utility functions
```

### Component Details
- **main.py**: Application initialization and configuration
- **app.py**: Streamlit interface and user interaction handling
- **code_assistant.py**: Core AI processing and task management
- **task_service.py**: Task-specific business logic
- **session_state.py**: Application state management

## Task Specifications

### Code Generation
Input: Natural language description  
Output: Production-ready Python code  
Features:
- Error handling implementation
- Type hint integration
- Documentation generation
- PEP standard compliance
- Performance optimization

### Code Review
Analysis Components:
- Code quality assessment
- Bug identification
- Performance analysis
- Security evaluation
- Optimization recommendations

### Documentation Generation
Output Components:
- System overview
- Implementation details
- API documentation
- Usage examples
- Parameter specifications

### Test Generation
Capabilities:
- Pytest framework integration
- Edge case coverage
- Error scenario testing
- Assertion implementation
- Test documentation

### Query Processing
Features:
- Context-aware responses
- Source citation
- Code comprehension
- Implementation guidance
- Best practice recommendations

## Security Scanner

### Scan Options
- **Quick Scan**: Basic security checks
- **Deep Scan**: Comprehensive analysis
- **Custom Scan**: Configure specific checks

### Input Methods
- **Upload Files**: Scan uploaded code files
- **Scan Code Snippet**: Direct code input for scanning
- **Git Repository**: Scan code from a Git repository with authentication support

### Scan Results
- Detailed vulnerability reports with severity filters
- Export options in JSON or PDF formats
- Historical scan data with trend analysis

## Configuration

### Environment Variables
```
GROQ_API_KEY          # Required for LLaMA 3
OPENAI_API_KEY        # Optional for GPT-4
LLAMA_CLOUD_API_KEY   # Optional for cloud services
```

### Model Configuration
- Embedding Model: BAAI/bge-small-en-v1.5
- Vector Store: LlamaIndex implementation
- Node Parser: SentenceSplitter configuration

## Usage Guidelines

### Task Selection
1. Choose task type from available options
2. Input task-specific requirements
3. Review generated output
4. Access saved files in output directory

### Document Management
1. Upload relevant files via UI
2. System automatically indexes content
3. Access indexed content via queries
4. Refresh index for updates

## Development

### Extension Points
- Model integration interface
- Task type implementation
- UI component development
- Service layer modification

### Best Practices
- Follow PEP standards
- Implement comprehensive error handling
- Maintain type hints
- Document new features
- Include unit tests
