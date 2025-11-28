""" EXPERIMENT 1 - This code uses a simple RAG mechanism with minimal prompts that retrieve relevant top_k snippets from input data layers and 
provides a code response. The input queries are provided in a .json format and the output code from LLM will also be 
generated in .json file called 'generated_code.json'. """

# IMPORTS
import json 
import os 
from dotenv import load_dotenv

from langsmith.run_helpers import traceable 
from langchain_ollama import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from retrieval_pipeline import retrieve_from_all_collections
from templates import get_template

PROJECT_PATH = r"C:\Users\medha\OneDrive - University of Twente\Documents\YEAR2\MasterThesis\Codes\MScThesis_1511_adjusted"
## LANGSMITH 
load_dotenv(dotenv_path = PROJECT_PATH + r"\.env")
# print(os.getenv("LANGSMITH_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "TestingRAG_System"

PERSIST_DIR = r"C:\Users\medha\OneDrive - University of Twente\Documents\YEAR2\MasterThesis\Codes\chroma_db"
COLLECTIONS = ["Python_docs", "input_layers"]

## CODE GENERATION -> NOW TRACKED BY LANGSMITH (20-11)
@traceable(name="baseline-code-generation")
def generate_code_for_query(query, template_name="baseline", persist_dir=PERSIST_DIR, collections=COLLECTIONS):
    template_info = get_template(template_name)
    template_str = template_info["template"]
    use_context = template_info["context"]
    use_rag_chain = template_info["use_rag_chain"]

    if use_context:
        context = retrieve_from_all_collections(
            query=query, 
            persist_dir=persist_dir, 
            collections=collections, 
            top_k=3
        )
    llm = OllamaLLM(model="llama3.2", temperature=0)

    # template = """"
    # You are a Python geospatial assistant that generates **accurate, minimal, and executable** code. 
    # Use retrieved context to guide your response. 

    # ---

    # ### Context:
    # {context}

    # ### User Query:
    # {query}

    # Output only Python code in a code block.
    # """
    if use_rag_chain:
        prompt = ChatPromptTemplate.from_template(template_str)
        rag_chain = (
            {
                "context": RunnablePassthrough() | (lambda q: context), 
                "query": RunnablePassthrough(), 
            }
            | prompt 
            | llm 
        )
        response = rag_chain.invoke(query)
    else:
        inputs = {"query": query}
        response = llm.invoke(prompt.format(**inputs))
    return response, context

### BATCH RUN 
def run_as_batch(input_json, output_json, template="baseline", **kwargs):
    with open(input_json, "r") as f:
        queries = json.load(f)
    
    results = []

    for item in queries:
        query = item["query"]
        print(f"[INFO] Processing query: {query}")

        llm_response, context = generate_code_for_query(
            query, 
            template_name=template,
            **kwargs
            )

        results.append({
            "query": query, 
            "retrieved_context": context,
            "response": llm_response,
        })
    
    with open(output_json, "w") as file:
        json.dump(results, file, indent=2)

    print(f"[SUCCESS] Saved generated outputs to: {output_json}")

if __name__ == "__main__":
    run_as_batch(PROJECT_PATH+r"\queries.json", PROJECT_PATH + r"\generated_code.json", template="simple_rag")