def validate_generated_code(user_query, code, ollama_model="llama3.2"):
    # ---- 1. Syntax Validation ----
    try:
        compile(code, "<string>", "exec")
        syntax_ok = True
    except Exception as e:
        return {"stage": "syntax", "verdict": "fail", "error": str(e)}

    # ---- 2. LLM-as-a-Judge ----
    judge_prompt = f"""
    You are a geospatial programming expert.

    Evaluate whether the following code correctly solves this task:
    Query: {user_query}

    Code:
    ```python
    {code}
    ```

    Evaluate the code and return JSON like this:
    {{
        "correctness_score": 0-5,
        "relevance_score": 0-5,
        "efficiency_score": 0-5,
        "safety_score": 0-5,
        "verdict": "pass" or "fail",
        "explanation": "one-sentence reason"
    }}
    """

    import ollama
    result = ollama.generate(model=ollama_model, prompt=judge_prompt)
    return json.loads(result["response"])
