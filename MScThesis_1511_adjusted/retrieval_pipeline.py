"""
Retrieves relevant context for geospatial query context creation and code generation. 
It searched across multiple Chroma collections: PyQGIS, Python libs, and user provided input layers 
"""
import os, json, re
from chromadb import PersistentClient 
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

def normalize_collection_name(name):
    """normalize PyQGIS collection name"""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def load_collections(client):
    """Return all available Chroma collection names"""
    try:
        return [c.name for c in client.list_collections()]
    except Exception as e:
        print(f"[WARN] Failed to list collections: {e}")
        return []


# def find_first_existing_collection(persist_dir, candidates):
#     """Given a list of candidate collection names, return the first one present in the DB.

#     Args:
#         persist_dir: path to chroma persistent client
#         candidates: iterable of collection name strings

#     Returns:
#         The first matching collection name, or None if none exist.
#     """
#     try:
#         client = PersistentClient(path=persist_dir)
#         all_collections = load_collections(client)
#         for c in candidates:
#             if c in all_collections:
#                 return c
#     except Exception as e:
#         print(f"[WARN] Failed to find existing collection: {e}")
#     return None
    
def summarize_docs(docs, limit=3):
    """Concatenate or summarize top documents for prompt context"""
    flat_docs = []
    for d in docs:
        if isinstance(d, list):
            flat_docs.extend(d)
        elif isinstance(d, str):
            flat_docs.append(d)
        else:
            flat_docs.append(str(d))
    summarized = "\n\n".join(flat_docs[:limit])
    return summarized if summarized else "No relevant documentation found."

# def retrieve_contexts(user_query, persist_dir, version="3.34", top_k=3, collections=None):
#     """
#     Retrieves relevant documentation from vectorstore collections:
#     python_docs
#     PyQGIS_docs 
#     User provided input layers metadata
#     """
#     client = PersistentClient(path=persist_dir)
#     embedder = SentenceTransformer("all-MiniLM-L6-v2")

#     q_emb = embedder.encode([user_query], convert_to_numpy=True).tolist()[0]

#     all_collections = load_collections(client)

#     results = {}

#     target_collections = collections if collections is not None else [
#         normalize_collection_name(f"pyqgis_{version.replace('.', '_')}_docs"),
#         "Python_docs",
#         "input_layers",
#     ]

#     for coll_name in target_collections:
#         if coll_name in all_collections:
#             try:
#                 coll = client.get_collection(coll_name)
#                 res = coll.query(query_embeddings=[q_emb], n_results=top_k, include=['documents', 'metadatas'])
#                 docs = res.get('documents', [[]])[0]
#                 if coll_name == 'input_layers':
#                     parsed = [json.loads(doc) if isinstance(doc, str) else doc for doc in docs]
#                     results[coll_name] = parsed
#                 else:
#                     results[coll_name] = docs
#             except Exception as e:
#                 print(f"[WARN] Failed to query collection {coll_name}: {e}")
#                 results[coll_name] = []
#         else:
#             print(f"[INFO] Collection not found: {coll_name}")
#             results[coll_name] = []

#     parts = []

#     pyqgis_key = normalize_collection_name(f"pyqgis_{version.replace('.', '_')}_docs")
#     parts.append(f"----PyQGIS Context---\n{summarize_docs(results.get(pyqgis_key, []))}\n\n")
#     parts.append(f"----Python docs Context----\n{summarize_docs(results.get('Python_docs', []))}\n\n")
#     parts.append(f"----Input Layer metadata----\n{json.dumps(results.get('input_layers', results.get('input_layers', [])), indent=2)}")

#     context_summary = "".join(parts)
#     return results, context_summary 

#### UNIFIED RETRIEVAL - ADJUSTED 01-11-2026
def retrieve_from_all_collections(query, persist_dir, collections=None, version="3.34", top_k=3):
    """
    Retrieves relevant context across specified Chroma collections.
    Falls back to default ones if none provided.
    Returns both structured results and a formatted text summary.
    """
    print(f"\n :)))Running retrieve_from_all_collections() with query='{query}'")
    print(f"Using persist_dir={persist_dir}")
    client = PersistentClient(path=persist_dir)
    
    # Using HuggingFace embeddings for consistency
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    q_emb = embedder.embed_query(query)
    
    all_collections = load_collections(client)

    # Default target collections
    if collections is None:
        collections = [
            normalize_collection_name(f"pyqgis_{version.replace('.', '_')}_docs"),
            "Python_docs",
            "input_layers",
        ]

    results = {}
    all_docs_text = []

    for coll_name in collections:
        if coll_name not in all_collections:
            print(f"[INFO] Collection not found: {coll_name}")
            results[coll_name] = []
            continue
        
        try:
            coll = client.get_collection(coll_name)
            res = coll.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas"])
            
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            
            # Handle input_layers specially (they’re JSON)
            if coll_name == "input_layers":
                parsed = [json.loads(d) if isinstance(d, str) else d for d in docs]
                results[coll_name] = parsed
            else:
                results[coll_name] = docs

            for doc, md in zip(docs, metas):
                doc_text = f"[{coll_name}] {doc}\nMetadata: {json.dumps(md, ensure_ascii=False)}"
                all_docs_text.append(doc_text)

        except Exception as e:
            print(f"[WARN] Could not query collection {coll_name}: {e}")
            results[coll_name] = []

    # Create a summary context for LLM input
    context_summary = "\n\n".join(all_docs_text[:top_k * len(collections)]) or "No relevant data retrieved."

    # return results, context_summary
    # 18-11-2025 -> Adjusting to make the function retrieve collections separately
    return {
        "input_layers": results.get("input_layers", []), 
        "python_docs": results.get("Python_docs", []), 
        "pyqgis_docs": results.get(normalize_collection_name(f"pyqgis_{version.replace('.', '_')}_docs"), [])
    }, context_summary

if __name__ == "__main__":
    import argparse 
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="User natural language query")
    p.add_argument("--presist_dir", default="./chroma_db", help="Chroma vectorstore path")
    p.add_argument("--version", default="3.34", help="QGIS version used for PyQGIS ingestion")
    p.add_argument("--collections", nargs="*", default=None, help="Specify which collection to search")
    args = p.parse_args()

    results, summary = retrieve_from_all_collections(args.query, args.persist_dir, version=args.version)
    print("\n[RETRIEVED CONTEXT SUMMARY]")
    print(summary)