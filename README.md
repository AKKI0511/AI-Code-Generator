## Overview

This project provides an intelligent code generation and documentation tool utilizing natural language processing. The tool is designed to read and analyze code files, generate new code based on user prompts, and provide detailed descriptions and filenames for the generated code. The implementation leverages the `llama_index` library and models from `Ollama`.

## Project Structure

The project consists of three main Python files:

1. **main.py**: The primary script that sets up the models, reads documents, and manages the user interaction loop.
2. **code_reader.py**: Defines a function to read code files and a tool to interface with the main script.
3. **prompts.py**: Contains context and template definitions used for generating prompts.

## Dependencies

- `llama_index`
- `pydantic`
- `dotenv`
- `os`
- `ast`

Make sure to install the necessary dependencies using the following command:
```bash
pip install llama_index pydantic python-dotenv
```

If you find any dependancy error install all dependancies from requirements.txt:
```bash
pip install -r requirements.txt
```

### NOTE: 
-  To use Ollama you will have to locally download it from here: https://ollama.com/
You can also use any other model from langchain (langchain-groq/langchain-openai).
-  You will also require an API key from llama-cloud: https://cloud.llamaindex.ai/login

## Configuration

Create a `.env` file in the root directory of the project to store environment variables. The `.env` file should contain any necessary configuration details such as API keys or other sensitive information.

## Usage

### 1. Initialize and Load Environment Variables

The script starts by loading environment variables from the `.env` file:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 2. Model Initialization

Initialize the `Ollama` models for document processing and code-related tasks:
```python
llm = Ollama(model="mistral", request_timeout=30.0)
code_llm = Ollama(model="codellama")
```

### 3. Document Reading and Processing

Load documents from the `./data` directory:
```python
documents = SimpleDirectoryReader("./data", file_extractor={".pdf": parser}).load_data()
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)
```

### 4. Define Tools

Define tools for the `ReActAgent`, including a query engine tool and a custom code reader:
```python
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="this gives documentation about code for an API. Use this for reading docs for the API",
        ),
    ),
    code_reader,
]
```

### 5. Agent and Output Parser Setup

Set up the `ReActAgent` and define the output parser using a Pydantic model:
```python
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])
```

### 6. Main Loop for User Interaction

Start the main loop to process user inputs and generate code:
```python
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            break
        except Exception as e:
            retries += 1
            print(f"Error occurred, retry #{retries}:", e)

    if retries >= 3:
        print("Unable to process request, try again...")
        continue

    print("Code generated")
    print(cleaned_json["code"])
    print("\n\nDescription:", cleaned_json["description"])

    filename = cleaned_json["filename"]

    try:
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filename)
    except:
        print("Error saving file…")
```

## File Details

### main.py
- **Function**: Sets up models, reads documents, manages tools, and processes user inputs to generate code.
- **Key Components**:
  - Loading environment variables.
  - Initializing LLM models.
  - Reading and processing documents.
  - Defining tools for the ReAct agent.
  - Setting up a user interaction loop to generate and save code.

### code_reader.py
- **Function**: Provides a tool to read and return the contents of code files.
- **Key Components**:
  - `code_reader_func(file_name)`: Reads the specified file and returns its content.
  - `code_reader`: A `FunctionTool` object that utilizes `code_reader_func`.

### prompts.py
- **Function**: Contains context and template definitions used for generating prompts.
- **Key Components**:
  - `context`: Describes the primary role of the agent.
  - `code_parser_template`: A template for parsing code generation responses into structured JSON.

## Running the Project

1. Ensure all dependencies are installed.
2. Place your documents in the `./data` directory.
3. Start the main script:
```bash
python main.py
```
4. Enter prompts to generate code or analyze existing code files. Enter `q` to quit the program.
