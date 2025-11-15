import streamlit as st 
import json 
import re
from info_gate import  run_information_gate, clarification_interaction
from langchain_ollama import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from retrieval_pipeline import retrieve_from_all_collections

PERSIST_DIR = r"C:\Users\medha\OneDrive - University of Twente\Documents\YEAR2\Master'sThesis\Codes\chroma_db"
COLLECTIONS = ["Python_docs", "input_layers"]

default_session = {
    "stage": "query",
    "user_query": "",
    "gate_result": None,
    "clarification": None,
    "context": None,
    "response": None
}
for key, value in default_session.items():
    if key not in st.session_state:
        st.session_state[key] = value

# dummy_input_layers = [
#     {"path": "data/landuse_rotterdam.shp", "type": "vector", "crs": "EPSG:28992", "fields": ["class_2018", "area"]},
#     {"path": "data/elevation_netherlands.tif", "type": "raster", "crs": "EPSG:28992"}
# ]
# dummy_retrieved_docs = [
#     "The area of Wetlands in Rotterdam can be calculated using the 'class_2018' field in the landuse dataset.",
#     "In PyQGIS, you can use the 'QgsVectorLayer' class to load vector data and the 'QgsRasterLayer' class for raster data."
# ]

st.set_page_config(page_title="GeoCode Assistant", layout="wide")
st.title("GeoCode Assistant")

st.markdown("""
This prototype is testing an Information Gate for geospatial code generation. 
Enter a geospatial query; the system checks for context, if it needs more information it asks for clarification.
After clarification, it proceeds to generate code with LLAMA3.2.
""")

st.sidebar.header("Settings")

## Will be adjusted as an LLM decision -> LLM decides the env and task type based on the query 
# env = st.sidebar.selectbox("Select Environment", ["Python", "PyQGIS"])
# task = st.sidebar.selectbox("Select Task", ["Spatial Analysis", "Data Extraction", "Visualization", "Other"])

# user_query = st.text_area("Enter your geospatial query:",
#                           placeholder="e.g. What is the area of the intersection of Polygon A and Polygon B")

# if st.button("Run Information Gate", type="primary"):
#     if not user_query.strip():
#         st.warning("Please enter a query first")
#     else:
#         with st.spinner("Running information gate..."):
#             gate_result = run_information_gate(
#                 user_query=user_query, 
#                 env=env,
#                 task=task, 
#                 input_layer_metas=dummy_input_layers, 
#                 retrieved_docs=dummy_retrieved_docs
#             )
#         st.subheader("Gate Decision")
#         st.json(gate_result)

#         if gate_result["need_clarification"]:
#             st.warning(f"Model requests clarification:\n\n**{gate_result['question']}**")
#             clarification = st.text_input("Your clarification:")
#             if clarification:
#                 st.success(f"Clarification received: {clarification}")
#             else:
#                 # Need to add patience mechanism, if user does not provide a clarification then the LLM proceeds to generation stage
#                 st.success("No clarification provided, the system will proceed")

# if st.button("Run Query"):
#     if not user_query.strip():
#         st.warning("Please enter a query before running.")
#         st.stop()

#     st.info("Running information gate...")
#     gate_result = run_information_gate(
#         user_query=user_query,
#         env=env,
#         task=task,
#         input_layer_metas=[],  # Will be replaced with actual user input layers
#         retrieved_docs=[]
#     )
#     st.subheader("Information Gate Result")
#     st.json(gate_result)

#     clarification = None 
#     if gate_result.get("need_clarification"):
#         st.warning(f"Model requests clarification: {gate_result['question']}")
#         clarification = st.text_input("Your answer (press Enter to continur):")
#     if clarification or not gate_result.get("need_clarification"):
#         st.info("Fetching context from collections...")
#         context = retrieve_from_all_collections(
#             user_query,
#             PERSIST_DIR, 
#             COLLECTIONS, 
#             top_k=3
#         )
#         st.text_area("Retrieved Context", context, height=200)

#         st.info("Generating geospatial code...")
#         llm = OllamaLLM(model="llama3.2", temperature=0)
#         template = """"
#         You are a Python geospatial assistant that generates accurate and efficient code.

#         Use the provided context to guide your answer.

#         ---
#         ### Context:
#         {context}

#         ### User Query:
#         {query}

#         ### Instructions:
#         1. Generate **complete, well-commented Python code**.
#         2. Use correct geospatial libraries (PyQGIS, Geopandas, Rasterio).
#         3. Add a short explanation after the code block.

#         ### Output Format:
#         ```python
#         # code here        
#         """
#         prompt = ChatPromptTemplate.from_template(template)
#         rag_chain = (
#             {
#                 "context": RunnablePassthrough() | (lambda q: context),
#                 "query": RunnablePassthrough()
#             }
#             | prompt
#             | llm
#         )

#         response = rag_chain.invoke(user_query)
#         st.subheader("Generated code and explanation")
#         st.markdown(response)

st.sidebar.header("Environment settings")
env = st.sidebar.selectbox("Select Environment", ["Python", "PyQGIS"])
task = st.sidebar.selectbox("Select Task", ["Spatial Analysis", "Data Extraction", "Visualization", "Other"])  

if st.session_state["stage"] == "query":
    user_query = st.text_area("Enter your geospatial query:",
                              placeholder="e.g. What is the area of the intersection of Polygon A and Polygon B")   
    if st.button("Run Query"):
        if not user_query.strip():
            st.warning("Please enter a query before running.")
            st.stop()

        with st.spinner("Running information gate..."):
            gate_result = run_information_gate(
                user_query=user_query, 
                env=env,
                task=task,
                input_layer_metas=[],
                retrieved_docs=[]
            )
        st.session_state["user_query"] = user_query
        st.session_state["gate_result"] = gate_result

        if gate_result.get("need_clarification"):
            st.session_state["stage"] = "clarification"
        else:
            st.session_state["stage"] = "generation"
        st.rerun()

elif st.session_state["stage"] == "clarification":
    gate_result = st.session_state["gate_result"]
    st.subheader("Clarification requested")
    st.write(gate_result["clarification_question"])
    clarification = st.text_input("Please provide clarification (press Enter to contune)")

    if st.button("Submit Clarification"):
        if not clarification:
            st.info("No clarification provided, the system will proceed.")
            clarification = "No clarification provided"
        st.session_state["clarification"] = clarification
        st.session_state["stage"] = "generation"
        st.rerun()

elif st.session_state["stage"] == "generation":
    st.subheader("Generating geospatial code...")
    user_query = st.session_state["user_query"]
    clarification = st.session_state.get("clarification", "")
    final_query = f"{user_query}\n\nClarification: {clarification}"

    if st.session_state["context"] is None:
        with st.spinner("Fetching context from collections..."):
            context = retrieve_from_all_collections(
                final_query,
                PERSIST_DIR, 
                COLLECTIONS, 
                top_k=3
            )
            st.session_state["context"] = context
    else:
        context = st.session_state["context"]
    # st.text_area("Retrieved Context", context, height=200)
    with st.expander("Retrieved context (click to expand)"):
        st.text_area("Retrieved Context", context, height=200)

    if st.session_state["response"] is None:
        with st.spinner("Generating geospatial code..."):
            llm = OllamaLLM(model="llama3.2", temperature=0)
            # template = """"
            # You are a Python geospatial assistant that generates accurate and efficient code.

            # Use the provided context and clarification to guide your answer.

            # ---
            # ### Context:
            # {context}

            # ### User Query:
            # {query}

            # ### Clarification:
            # {clarification}

            # ### Instructions:
            # 1. Generate **complete, well-commented Python code**.
            # 2. Use correct geospatial libraries (PyQGIS, Geopandas, Rasterio).
            # 3. Add a short explanation after the code block.

            # You MUST output ONLY a python code block in this format
            # ### Output Format:
            # ```python
            # # code here  
            # ```      
            # """
        ##========ADJUSTED ON 13-11-2025 (Adding few-shot examples)============= (switching accurate and syntacically correct to **accurate minimala and executable**)
            template = """
            You are a Python geospatial assistant that generates **accurate, minimal, and executable** code. 
            Use the provided context and clarification to guide your answer.

            ---

            ### Context:
            {context}

            ### User Query:
            {query}

            ### Clarification (if any):
            {clarification}

            ### Available Examples:
            Use the following examples as a guide for syntax, library use, and logical structure:
            **Examples**
            1. Load shapefile:
            ```python
            import geopandas as gpd
            shp = gpd.read_file('file.shp')
            print(shp.head())
            2. Load specific layer from a GeoPackage 
            ```python 
            import geopandas as gps 
            gdf = gpd.read_file("data/urban.gpkg", layer="roads")
            print(gdf.head())
            3. Inspect available layers in a GeoPackage 
            import fiona 
            layers = fiona.listlayers('data/urban.gpkg')
            print(layers)
            """


            prompt = ChatPromptTemplate.from_template(template)
            rag_chain = (
                {
                    "context": RunnablePassthrough() | (lambda q: context),
                    "query": RunnablePassthrough(), 
                    "clarification": RunnablePassthrough()
                }
                | prompt
                | llm
            )

            response = rag_chain.invoke(final_query)
            st.session_state["response"] = response
    else:
        response = st.session_state["response"]
    st.subheader("Generated code and explanation")
    # st.markdown(response)
    # Display code as formatted code block 07-11-2025
    # st.markdown(response, unsafe_allow_html=True)
    code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
    if code_match:
        code_block = code_match.group(1).strip()
        st.code(code_block, language="python")
    else:
        st.warning("No code block found in the response.")
    # Explanation 
    explanation = re.sub(r"```python(.*?)```", "", response, flags=re.DOTALL).strip()
    if explanation:
        st.markdown("### Explanation")
        st.write(explanation)


    if st.button("Start New Query"):
        for key in ["stage", "gate_result", "clarification", "context", "response"]:
            st.session_state[key] = None 
        st.session_state["stage"] = "query"
        st.rerun()
        
        