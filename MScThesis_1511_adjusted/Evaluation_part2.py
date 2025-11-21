"""This evaluation setup uses LLM-as-a-judge to assess the semantic similarity of code"""
import json 
import re
from langchain_ollama import OllamaLLM 

def llm_judge_evaluate(ref_code:str, gen_code:str, model_name="llama3.2"):
    """
    This function uses LLM-as-a-judge to determine the semantic equivalence and quality of the generated code. 
    Returns a JSON dictionary with evaluation scores + reasoning.
    """

    llm = OllamaLLM(model=model_name, temperature=0)

    prompt = f"""
You are a Python code evaluation assistant. 

Your job is to compare a *reference solution* with a *generated solution* and evaluate:
1. **Semantic correctness** (Do they compute the same thing?)
2. **Completeness** (Does generated code fully solve the task?)
3. **Code quality** (Structure, clarity, best practices)
4. **Equivalence** (true/false)

Respond ONLY in valid JSON with this structure:
{{
"equivalent": true|false, 
"semantic_score": 0.0 to 1.0, 
"completeness_score": 0.0 to 1.0, 
"quality_score": 0.0 to 1.0, 
"final_score": "short explanation"
}}

Reference Code:
{ref_code}

Generated Code:
{gen_code}
"""
    response = llm.invoke(prompt)
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return{
            "equivalent": False, 
            "semantic_score": 0.0, 
            "completeness_score": 0.0, 
            "quality_score": 0.0, 
            "final_score": 0.0, 
            "reasoning":"LLM returned non-JSON output."
        }
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "equivalent": False, 
            "semantic_score": 0.0, 
            "completeness_score": 0.0, 
            "quality_score": 0.0, 
            "final_score": 0.0, 
            "reasoning": "LLM JSON parse error"
        }
    

# Example testing 
# gen_code = """
# import geopandas as gpd
# df = gpd.read_file("data/spoorwegen.gpkg", layer="roads")
# multi_line_gdf = df[df.geometry.type == 'MultiLineString']
# length = multi_line_gdf.geometry.length.sum()
# print(f'Total length: {length} meters')
# """

# ref_code = """
# import geopandas as gpd
# gdf = gpd.read_file("inputdata/spoorwegen.gpkg", layer="spooras")
# total_length = gdf.geometry.length.sum()
# print(f'The total length of railway lines in the Netherlands is {total_length} meters')
# """
gen_code = """"
import geopandas as gpd

# Load shapefile
shp = gpd.read_file('inputdata\\NL_Provinces.shp')

# Filter the provinces layer for 'Overijssel'
overijssel = shp[shp['NAME_1'] == 'OVERIJSEL']

# Perform an overlay operation to find neighboring provinces
neighbors = overijssel[shp.geometry.intersects(overijssel.geometry)]

print(neighbors)
"""


ref_code = """
import geopandas as gpd

# Load shapefile
shp = gpd.read_file('inputdata\\NL_Provinces.shp')

# Filter the provinces layer for 'Overijssel'
overijssel = shp[shp['NAME_1'] == 'Overijssel'].iloc[0]

# Get its geometry 
ov_geom = overijssel.geometry

# Find the neighbors: Provinces that touch Overijssel 
neighbors = shp[shp.geometry.touches(ov_geom)]

print(neighbors[['NAME_1']])
"""





result = llm_judge_evaluate(ref_code, gen_code)
print(result)