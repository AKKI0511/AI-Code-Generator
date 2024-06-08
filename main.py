from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template
from code_reader import code_reader
from dotenv import load_dotenv
import os
import ast

load_dotenv()

# Initialize the Ollama model with specific configurations.
llm = Ollama(model="mistral", request_timeout=30.0)

# Set up a parser for processing documents.
parser = LlamaParse(result_type="markdown")

# Define a file extractor for PDF documents using the previously defined parser.
file_extractor = {".pdf": parser}
# Load documents from a directory, processing supported file types.
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# Resolve and load the embedding model for document indexing.
embed_model = resolve_embed_model("local:BAAI/bge-m3")
# Create a vector index from the loaded documents using the embedding model.
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
# Convert the vector index into a query engine using the Ollama model.
query_engine = vector_index.as_query_engine(llm=llm)

# Define tools for the ReAct Agent, including a query engine tool and a custom code reader.
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

# Initialize another Ollama model specifically for code-related tasks.
code_llm = Ollama(model="codellama")
# Create a ReAct Agent with the defined tools, context, and verbosity settings.
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

# Define a Pydantic model for structured output from the code generation process.
class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

# Set up an output parser using the Pydantic model to format the code generation output.
parser = PydanticOutputParser(CodeOutput)
# Generate a JSON prompt string from a template and format it using the parser.
json_prompt_str = parser.format(code_parser_template)
# Create a prompt template from the JSON prompt string.
json_prompt_tmpl = PromptTemplate(json_prompt_str)
# Define a query pipeline that uses the prompt template and the Ollama model.
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

# Main loop for processing user input and generating code.
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            # Query the ReAct Agent with the user's prompt and process the result through the output pipeline.
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            # Clean the JSON output, removing any unwanted prefixes.
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}:", e)

    if retries >= 3:
        print("Unable to process request, try again...")
        continue

    # Display the generated code and its description.
    print("Code generated")
    print(cleaned_json["code"])
    print("\n\nDesciption:", cleaned_json["description"])

    # Save the generated code to a file.
    filename = cleaned_json["filename"]

    try:
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filename)
    except:
        print("Error saving file…")