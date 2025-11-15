import json, os 
from langchain_ollama import OllamaLLM 
from sentence_transformers import SentenceTransformer 
from chromadb import PersistentClient 
import re

DEBUG = True 

def format_layer_metadata_for_prompt(metas):
    lines = []
    for m in metas:
        lines.append(
            f"- {m.get('path', 'unknown')} | type: {m.get('type')} | crs:{m.get('crs')} | fields: {m.get('fields', [])}"
        )
    return "\n".join(lines) if lines else "No usre input layers provided"

def format_docs_summary(docs):
    flat_docs = []
    for d in docs:
        if isinstance(d, list):
            flat_docs.extend(d)
        elif isinstance(d, str):
            flat_docs.append(d)
        else:
            flat_docs.append(str(d))
    return "\n".join(flat_docs[:5])  ##<=truncate to reduce prompt size if needed]

# Main information gate 

# def run_information_gate(user_query, env, task, input_layer_metas, retrieved_docs, model_name="llama3.2"):
#     """
#     [NEED TO WRITE DOCSTRING]
#     """
#     llm = OllamaLLM(model=model_name, temperature=0)
#     prompt = f"""
#     You are a geospatial reasoning assistant.
#     You must check if the query can be executed using the available data and documentation. Identify whether clarification is needed about:
#     - "Missing or ambiguous datasets (input layers, attributes, geometry type)"
#     - "Processing parameters (CRS, buffer units, aggregation levels)"
#     - "Output preferences (format, map, table)"
#     Ask exactly one concise clarification question that would help resolve the ambiguity
#     Output STRICT JSON:
#     {{
#     "need_clarification": true|false,
#     "reasoning": "short explanation of what is missing or ambiguous"
#     "question": "one concise question for the user (only if need_clarification=true)",
#     "required_fields": ["optional", "list", "of", "expected", "metadata", "fields"],
#     "confidence": float
#     }}

#     ### User Query:
#     {user_query}

#     ### Environment:
#     {env} | Task: {task}

#     ### Input Layer Metadata:
#     {format_layer_metadata_for_prompt(input_layer_metas)}

#     ### Retrieved Context:
#     {format_docs_summary(retrieved_docs)}
#     """ 
#     if DEBUG:
#         print("\n[INFO GATE] ===== Prompt Sent to LLM =====")
#         print(prompt)
#         print("=========================================\n")

# 31-10-2025 16:06
# In the current DEBUG Mode, the if the LLM has a clarification question, need_clarification is set to True
# Earlier the system respected the LLM output of need_clarification=False, even when it had a question
    # response = llm.invoke(prompt)

    # if DEBUG:
    #     print("\n[INFO GATE] ===== Raw Model Response =====")
    #     print(response)
    #     print("=========================================\n")
    # try:
    #     clean_response = response.strip()
    #     if "```" in clean_response:
    #         clean_response = clean_response.split("```")[-1]
    #     result = json.loads(clean_response)
    #     need_clar = bool(result.get("need_clarification", False))
    #     question = result.get("question") if isinstance(result.get("question"), str) else (str(result.get("question")) if result.get("question") is not None else "")
    #     required_fields = result.get("required_fields") if isinstance(result.get("required_fields"), list) else (list(result.get("required_fields")) if result.get("required_fields") is not None and isinstance(result.get("required_fields"), (tuple, set)) else [])
    #     try:
    #         confidence = float(result.get("confidence", 0.0))
    #     except Exception:
    #         confidence = 0.0

    #     if not need_clar and (question.strip() or required_fields):
    #         if DEBUG:
    #             print("[INFO GATE] Model output inconsistent: question/required_fields present but need_clarification is False. Forcing need_clarification=True.")
    #         need_clar = True

    #     result = {
    #         "need_clarification": need_clar,
    #         "question": question.strip(),
    #         "required_fields": required_fields,
    #         "confidence": confidence,
    #     }
    # except Exception as e:
    #     if DEBUG:
    #         print("[INFO GATE] JSON parsing failed:", e)
    #         print("[INFO GATE] Raw output:", response)
    #     result = {
    #         "need_clarification": True,
    #         "question": "Could you please clarify which dataset or attribute to use?",
    #         "required_fields": ["file_path"],
    #         "confidence": 0.0
    #     }

    # return result 



#====================ADJUSTMENT====================================
# 13-11-2025 
# Adjusting run_information_gate() function; Right now the function checks if the pre-conditions and post-conditions are satisfied 
# before proceeding to the generation steps
def run_information_gate(user_query, env, task, input_layer_metas, retrieved_docs, model_name="llama3.2"):
    """
    Evaluate whether the system has sufficient information to proceed with code genertion. 
    Checks pre-conditions and post-conditions. 
    Returns JSON decision 
    """
    llm = OllamaLLM(model=model_name, temperature=0)
    prompt=f"""
You are a geospatial reasoning and validation assistant. 
Your task is to check whether the user's query can be executed correctly 
based on the available data and context.

---
### Inputs 
User Query:
{user_query}

Environment: {env}
Task Type: {task}

Input Layers Metadata:
{format_layer_metadata_for_prompt(input_layer_metas)}

Retrieved_context:
{format_docs_summary(retrieved_docs)}

---
### Validation Logic:
You must check two things:
1. **Pre-conditions** - Are all necessary inputs provided or deliverable?
2. **Post-conditions** - Can a valid, meaningful output be produced?

If *both* are satisfied, return:
    preconditions_satisfied: true
    postconditions_satisfied: true 
    need_clarification: false

If not, return:
    need_clarification: true
    and a single clarification question to make the missing condition TRUE.

---

### Output Format (STRICT JSON)
{{
"preconditions_satisfied": true|false, 
"postconditions_satisfied": true|false, 
"need_clarification": true|false, 
"missing_elements": ["list of missing inputs or ambiguities"], 
"clarification_question": "Ask one concise question to resolve ambiguity", 
"confidence": float (0.0 to 1.0)
}}
"""

    if DEBUG:
        print("\n[INFO GATE PROMPT] ===")
        print(prompt)
        print("==============\n")

    response = llm.invoke(prompt)

    if DEBUG:
        print("\n[RAW MODEL RESPONSE]")
        print(response)
        print("=================\n")

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        print("[WARN] Could not parse JSON from model output.")
        return {
            "preconditions_satisfied": False, 
            "postconditions_satisfied": False, 
            "need_clarification": True, 
            "missing_elements": ["unknown"], 
            "clarification_question": "Could you specify which dataset or field to use?",
            "confidence": 0.0
        }
    try:
        result = json.loads(match.group(0))
        if not isinstance(result.get("need_clarification"), bool):
            result["need_clarification"] = bool(result.get("clarification_question"))
        return result
    except Exception as e:
        print(f"[ERROR] Failed to decode model output: {e}")
        return {
            "preconditions_satisfied": False, 
            "postcoditions_satified": False, 
            "need_clarification": True, 
            "missing_elements": ["unknown"], 
            "clarification_question": "Could you specify missing input(s)?", 
            "confidence": 0.0
        }







def clarification_interaction(gate_result):
    """Interactively ask user once for clarification"""
    if not gate_result.get("need_clarification"):
        return None 
    
    print(f"[INFO] Clarification reqd  ({gate_result.get('reasoning','')})")
    print(gate_result['clarification_question'])
    answer = input("Your answer (press Enter to skip):").strip()
    if not answer:
        answer = "proceed"

    clarification_data = {
        "clarification_answer": answer, 
        "reasoning": gate_result.get("reasoning", ""),
        "required_fields": gate_result.get("required_fields"), 
        "confidence":gate_result.get("confidence")
    }
    return clarification_data