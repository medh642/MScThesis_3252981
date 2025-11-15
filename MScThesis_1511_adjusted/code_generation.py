from info_gate import run_information_gate, clarification_interaction
from retrieval_pipeline import retrieve_from_all_collections

persist_dir = r"C:\Users\medha\OneDrive - University of Twente\Documents\YEAR2\Master'sThesis\Codes\chroma_db"
user_query = "What is the area of Wetlands landuse in attribute class_2018 in Rotterdam?"


version = "3.34"
pyqgis_coll = f"pyqgis_{version.replace('.', '_')}_docs"
collections = [pyqgis_coll, "Python_docs", "input_layers"]
print("Querying collections:", collections)

retrieved, summary = retrieve_from_all_collections(user_query, persist_dir, version=version, top_k=3, collections=collections)
input_layers_meta = retrieved.get('input_layers', [])
pyqgis_docs = retrieved.get(pyqgis_coll, [])
python_docs = retrieved.get('Python_docs', [])

gate_result = run_information_gate(
    user_query,
    env="Python",
    task="area_calculation",
    input_layer_metas=input_layers_meta,
    retrieved_docs=pyqgis_docs + python_docs
)

clarification = clarification_interaction(gate_result)
print("Clarification:", clarification)