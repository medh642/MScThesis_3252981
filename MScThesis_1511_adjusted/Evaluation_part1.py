"""This file provides a deterministic evaluation set up that tests the syntactic validity of the code generated
This evaluation does not consider the LLM-as-a-judge set up"""
from sklearn.metrics import precision_score, recall_score, f1_score 
import re 
import ast 
from difflib import SequenceMatcher
import traceback
# import astor -> lib needs to be installed in masterthesis_venv

def tokenize_code(code):
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\S", code)
    return tokens 

def compute_token_metrics(reference_code, generated_code):
    ref_tokens = tokenize_code(reference_code)
    gen_tokens = tokenize_code(generated_code)

    unique_tokens = list(set(ref_tokens + gen_tokens))

    ref_vec = [1 if t in ref_tokens else 0 for t in unique_tokens]
    gen_vec = [1 if t in gen_tokens else 0 for t in unique_tokens]

    precision = precision_score(ref_vec, gen_vec, zero_division=0)
    recall = recall_score(ref_vec, gen_vec, zero_division=0)
    f1 = f1_score(ref_vec, gen_vec, zero_division=0)

    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "num_ref_tokens": len(ref_tokens), 
        "num_gen_tokens": len(gen_tokens)
    }

gen_code = """
import geopandas as gpd
df = gpd.read_file("data/spoorwegen.gpkg", layer="roads")
multi_line_gdf = df[df.geometry.type == 'MultiLineString']
length = multi_line_gdf.geometry.length.sum()
print(f'Total length: {length} meters')
"""

ref_code = """
import geopandas as gpd
gdf = gpd.read_file("inputdata/spoorwegen.gpkg", layer="spooras")
total_length = gdf.geometry.length.sum()
print(f'The total length of railway lines in the Netherlands is {total_length} meters')
"""

# ref_code = """
# import math 
# radius = 10 
# area = math.pi * radius * radius
# """
# gen_code = """
# import math 
# r = 10 
# area = math.pi * r ** 2
# """


# def ast_similarity(reference_code, generated_code):
#     try:
#         ref_ast = ast.dump(ast.parse(reference_code))
#         gen_ast = ast.dump(ast.parse(generated_code))
#     except Exception:
#         return {"ast_similarity": 0.0}
#     score = SequenceMatcher(None, ref_ast, gen_ast).ratio()
#     return {"ast_similarity": score}

# print(ast_similarity(ref_code, gen_code))

class Normalizer(ast.NodeTransformer):
    """
    Replace identifier names (variables, args, function names, attribute names)
    with canonical placeholders VAR1, VAR2, ... to ignore naming differences.
    """
    def __init__(self):
        super().__init__()
        self.name_map = {}
        self.counter = 0

    def _canon(self, original):
        if original not in self.name_map:
            self.counter += 1
            self.name_map[original] = f"VAR{self.counter}"
        return self.name_map[original]

    # Replace simple names (variables, references)
    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
            new_id = self._canon(node.id)
            return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)
        return node

    # Replace argument names in function defs
    def visit_arg(self, node):
        if node.arg:
            node.arg = self._canon(node.arg)
        return node

    # Replace function names
    def visit_FunctionDef(self, node):
        if node.name:
            node.name = self._canon(node.name)
        # normalize decorator names and body
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        if node.name:
            node.name = self._canon(node.name)
        self.generic_visit(node)
        return node

    # Replace class names
    def visit_ClassDef(self, node):
        if node.name:
            node.name = self._canon(node.name)
        self.generic_visit(node)
        return node

    # Replace attribute names (object.attr -> object.VARx)
    def visit_Attribute(self, node):
        # normalize attribute identifier but keep the value (object) normalized by recursion
        node.value = self.visit(node.value)
        if isinstance(node.attr, str):
            node.attr = self._canon(node.attr)
        return node

    # Optionally normalize keywords (e.g., kwargs names)
    def visit_keyword(self, node):
        if node.arg:
            node.arg = self._canon(node.arg)
        node.value = self.visit(node.value)
        return node

    # Normalize aliases in import as: "import x as y" -> alias name becomes canonical
    def visit_alias(self, node):
        if node.asname:
            node.asname = self._canon(node.asname)
        else:
            # optionally canonicalize the imported name as well
            node.name = node.name  # keep original module path
        return node

def normalize_and_dump(code, debug=False):
    """
    Parse code to AST, normalize identifiers and return a stable AST string dump.
    Returns (dump_str, error_message). If successful error_message is None.
    """
    if not isinstance(code, str):
        return None, "code is not a string"

    # Quick guard: strip prompts/backticks that may be present in LLM outputs
    # remove fenced code blocks if present
    if "```" in code:
        # pick content inside last code fence or remove fences
        try:
            parts = code.split("```")
            # if there are code fences, prefer the last block with content
            inner = max(parts, key=len)
            code = inner
        except Exception:
            pass
    # Strip leading/trailing whitespace
    code = code.strip()
    if not code:
        return None, "empty code after stripping"

    try:
        tree = ast.parse(code)
    except Exception as e:
        tb = traceback.format_exc()
        if debug:
            print("[normalize_and_dump] ast.parse failed:", e)
            print(tb)
        return None, f"parse_error: {e}"

    try:
        normalizer = Normalizer()
        norm_tree = normalizer.visit(tree)
        ast.fix_missing_locations(norm_tree)
        dump = ast.dump(norm_tree, include_attributes=False)
        return dump, None
    except Exception as e:
        tb = traceback.format_exc()
        if debug:
            print("[normalize_and_dump] normalization failed:", e)
            print(tb)
        return None, f"normalize_error: {e}"

def ast_structural_similarity_normalized(ref_code, gen_code, debug=False):
    """
    Returns a float in [0,1] representing AST structural similarity after normalization.
    Also returns an optional dict with diagnostics when debug=True.
    """
    diagnostics = {}
    ref_dump, ref_err = normalize_and_dump(ref_code, debug=debug)
    gen_dump, gen_err = normalize_and_dump(gen_code, debug=debug)

    diagnostics['ref_err'] = ref_err
    diagnostics['gen_err'] = gen_err

    if ref_err or gen_err:
        # If parse/normalize failed, return 0 and diagnostics
        if debug:
            print("[ast_structural_similarity_normalized] Problems found:")
            print(" ref_err:", ref_err)
            print(" gen_err:", gen_err)
        return 0.0, diagnostics

    # Use SequenceMatcher ratio for structural similarity
    try:
        score = SequenceMatcher(None, ref_dump, gen_dump).ratio()
    except Exception as e:
        if debug:
            print("[ast_structural_similarity_normalized] SequenceMatcher error:", e)
        return 0.0, diagnostics

    diagnostics['ref_dump_sample'] = ref_dump[:300]
    diagnostics['gen_dump_sample'] = gen_dump[:300]
    diagnostics['score'] = score
    return score, diagnostics

# if __name__ == "__main__":
#     ref = """
# def compute_area(g):
#     return g.area
# """
#     gen = """
# def calc_area(poly):
#     return poly.area
# """
#     score, diag = ast_structural_similarity_normalized(ref, gen, debug=True)
#     print("Score:", score)
#     print("Diagnostics:", diag)
print(ast_structural_similarity_normalized(ref_code, gen_code, debug=True))
    
    

