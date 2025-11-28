from .baseline import BASELINE_TEMPLATE
from .simple_rag import SIMPLE_RAG_TEMPLATE

TEMPLATES = {
    "baseline": {
        "template": BASELINE_TEMPLATE, 
        "use_context": False,
        "use_rag_chain": False,
    }, 
    "simple_rag": {
        "template": SIMPLE_RAG_TEMPLATE, 
        "use_context": True,
        "use_rag_chain": True,
    }
}

def get_template(name: str):
    if name not in TEMPLATES:
        raise ValueError(f"Template '{name}' does not exist.")
    return TEMPLATES[name]