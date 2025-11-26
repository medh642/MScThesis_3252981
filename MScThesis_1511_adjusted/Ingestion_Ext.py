""" Latest
This file creates a knowledge base for the model to retrieve relevant information to guide code generation. 
It uses prompts.json (prompts_pyqgis and prompts_python), and scrapes from the PyQGIS documentation and important geospatial python libraries to access documentation of useful functions and tools.
The information is then converted into chunks (chunking + indexing) and then stored into a vectorDB (ChromaDB)
"""

# Import modules 
import os, re, time, json, hashlib, argparse
from urllib.parse import urljoin
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
import fiona
from chromadb.config import Settings 
from chromadb import PersistentClient 
import importlib, inspect
# For input layer ingestion 
import geopandas as gpd
import rasterio 
# For metadata enrichment with LLM 
from langchain_ollama import OllamaLLM 
import json, re 
from chromadb import PersistentClient 
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# Define constants 
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_WORKERS = 6
RATE_LIMIT_SECONDS = 1.0 
USER_AGENT = "pyqgis-ingestor/1.0 (+https://example.org)"
LAST_CRAWLED_PATH = "last_crawled.json"

# --------------------- UTILS ---------------------
def clean_text(string):
    if not string:
        return ""
    string = re.sub(r'\s+', ' ', string.replace("\xa0", " ")).strip()
    return string

def make_id(*parts):
    raw = "::".join(map(str, parts))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]

def load_cache(path=LAST_CRAWLED_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache, path=LAST_CRAWLED_PATH):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

def rate_limit():
    time.sleep(RATE_LIMIT_SECONDS)

# --------------------- SCRAPING ---------------------
def get_cookbook_urls(base_url):
    r = requests.get(base_url, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if href.startswith(base_url) and href.endswith(".html"):
            links.add(href)
    return sorted(list(links))

def fetch_page(url, cache, version="3.34", module=None, tags=None):
    """
    Fetch a page, respecting cache (ETags) and collect metadata for vector DB ingestion.
    """
    if cache is None:
        cache = {}

    headers = {"User-Agent": USER_AGENT}
    etag = cache.get(url, {}).get("etag")
    if etag:
        headers["If-None-Match"] = etag

    rate_limit()

    try:
        r = requests.get(url, headers=headers, timeout=20)
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return {}

    if r.status_code == 304:
        return {}
    if r.status_code != 200:
        print(f"[WARN] Skipping {url} (status {r.status_code})")
        return {}

    cache[url] = {
        "etag": r.headers.get("ETag"),
        "last_modified": r.headers.get("Last-Modified"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    soup = BeautifulSoup(r.text, "html.parser")
    title_tag = soup.find("h1") or soup.title
    title_text = title_tag.get_text(strip=True) if title_tag else url
    headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3", "h4"])]

    return {
        "url": url, 
        "soup": soup,
        "title": title_text,
        "headings": headings,
        "module": module,
        "version": version,
        "tags": tags or []
    }

# --------------------- PARSING ---------------------
def extract_sections(soup):
    body = soup.find("body") or soup
    main = body.find(attrs={"role": "main"}) or soup
    headers = main.find_all(re.compile("^h[1-6]$"))
    sections = []

    if not headers:
        text = clean_text(main.get_text())
        if text:
            sections.append({"heading": "Page", "text": text})
        return sections

    for header in headers:
        heading = header.get_text(strip=True)
        content_parts = []
        for sib in header.next_siblings:
            if getattr(sib, "name", None) and re.match("^h[1-6]$", sib.name):
                break
            txt = sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else ""
            if txt:
                content_parts.append(txt)
        content = clean_text(" ".join(content_parts))
        if content:
            sections.append({"heading": heading, "text": content})
    return sections

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i + size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def infer_module(url):
    if "vector" in url:
        return "qgis.core.vector"
    if "raster" in url:
        return "qgis.core.raster"
    if "processing" in url:
        return "qgis.processing"
    if "gui" in url:
        return "qgis.gui"
    return "pyqgis"

# --------------------- INGESTION ---------------------
def ingest(version, persist_dir, prompt_examples=None):
    os.makedirs(persist_dir, exist_ok=True) # New addition 
    base = f"https://docs.qgis.org/{version}/en/docs/pyqgis_developer_cookbook/"
    cache_path = os.path.join(persist_dir, "last_crawled.json")

    # Load cache safely
    cache = load_cache(cache_path)
    urls = get_cookbook_urls(base)
    print(f"[INFO] Found {len(urls)} pages")

    pages = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [
            ex.submit(fetch_page, u, cache, version=version, module="PyQGIS", tags=["cookbook"])
            for u in urls
        ]
        for fut in as_completed(futs):
            page = fut.result()
            if page:
                pages.append(page)

    save_cache(cache, cache_path)
    print(f"[INFO] Fetched {len(pages)} updated pages")

    # Prepare text chunks
    docs, metas, ids = [], [], []
    for p in pages:
        for sidx, sec in enumerate(extract_sections(p["soup"])):
            for cidx, chunk in enumerate(chunk_text(sec["text"])):
                doc_id = make_id(version, p['url'], sidx, cidx)
                docs.append(chunk)
                metas.append({
                    "url": p["url"],
                    "page_title": p["title"],
                    "heading": sec["heading"],
                    "module": infer_module(p["url"]),
                    "version": version,
                    "etag": cache.get(p["url"], {}).get("etag"),
                    "last_modified": cache.get(p["url"], {}).get("last_modified"),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "tags": ""
                })
                ids.append(doc_id)
    # New adjustment 
    def sanitize_metadata(meta):
        clean_meta = {}
        for k, v in (meta or {}).items():
            if v is None:
                clean_meta[k] = "" # Replacing None with empty string
            elif isinstance(v, (str, int, float, bool)) or v is None:
                clean_meta[k] = v
            elif isinstance(v, (list, dict)):
                clean_meta[k] = json.dumps(v, ensure_ascii=False)
            else:
                clean_meta[k] =str(v)
        return clean_meta
    metas = [sanitize_metadata(m) for m in metas if m is not None]

    print(f"[INFO] Prepared {len(docs)} chunks")

    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # embeddings = model.encode(docs, convert_to_numpy=True).tolist() 
    ### Changing to HuggingFaceEmbeddings to align with retrieval code
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.embed_documents(docs)


    # settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
    # client = chromadb.Client(settings)
    client = PersistentClient(path=persist_dir)
    collection_name = f"pyqgis_{version.replace('.', '_')}_docs"
    coll = client.get_or_create_collection(name=collection_name, metadata={"version": version})
    coll.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
    # client.persist()
    print(f"[INFO] Ingeste_d {len(docs)} chunks into {collection_name}")



    # --------------------- PROMPT EXAMPLES ---------------------
    if prompt_examples and os.path.exists(prompt_examples):
        with open(prompt_examples, "r", encoding="utf-8") as f:
            examples = json.load(f)

        ex_docs, ex_meta, ex_ids = [], [], []
        for ex in examples:
            text = f"PROMPT:\n{ex['prompt']}\n\nRESPONSE:\n{ex['response']}"
            # Adjusting dt error
            tags = ex.get("tags", [])
            if isinstance(tags, list):
                tags = ", ".join(tags)
            ex_docs.append(text)
            ex_meta.append(sanitize_metadata({
                "version": version,
                "tags": ex.get("tags", []),
                "source": "prompt_examples"
            }))
            ex_ids.append(str(ex.get("id", make_id('ex', ex['prompt'][:20]))))

        emb = embedder.embed_documents(ex_docs)
        ex_coll = client.get_or_create_collection(name=f"pyqgis_{version.replace('.', '_')}_examples")
        ex_coll.add(documents=ex_docs, embeddings=emb, metadatas=ex_meta, ids=ex_ids)
        # client.persist()
        print(f"[INFO] Added {len(ex_docs)} examples")

# ------------------ PARSING Python LIBRARIES-------------
def extract_module_docs(lib_name):
    """Extracts docstrings from Python geospatial libraries"""
    try:
        mod = importlib.import_module(lib_name)
    except ImportError:
        print(f"Could not import {lib_name}, skipping.")
        return []
    docs = []
    for name, obj in inspect.getmembers(mod):
        if inspect.isfunction(obj) or inspect.isclass(obj):
            doc = inspect.getdoc(obj)
            if doc:
                docs.append({
                    "name":f"{lib_name}.{name}",
                    "doc":doc
                })
    return docs 

def ingest_python_geo_docs(persist_dir, libs=None):
    """"
    This ingest block takes the docstrings from common geospatial Python libraries (Geopandas, Shapely, Rasterio) unless specified otherwise, 
    and stored them in a ChromaDB 
    """
    if libs is None:
        libs = ["geopandas", "shapely", "rasterio", "fiona", "folium"]
    
    os.makedirs(persist_dir, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = PersistentClient(path=persist_dir)

    all_docs, all_meta, all_ids = [], [], []

    for lib in libs:
        print(f"Extracting documentation from {lib}")
        extracted = extract_module_docs(lib)
        for e in extracted:
            chunk = chunk_text(e["doc"])
            for cidx, chunk in enumerate(chunk):
                doc_id = make_id(lib, e["name"], cidx)
                all_docs.append(chunk)
                all_meta.append({
                    "library":lib, 
                    "object_name":e["name"], 
                    "retrieved_at":datetime.now(timezone.utc).isoformat()
                })
                all_ids.append(doc_id)

        if not all_docs:
            print("[WARN] No documentation extracted")
            return 
        
        print(f"[INFO] Prepared {len(all_docs)} chunks from {len(libs)} libraries.")

        embeddings = model.encode(all_docs, convert_to_numpy=True).tolist()
        coll = client.get_or_create_collection("Python_docs")
        coll.add(documents=all_docs, embeddings=embeddings, metadatas=all_meta, ids=all_ids)
        print(f"[INFO] Ingested {len(all_docs)} chunks into Python_docs.")


# -------------------- USER INPUT LAYER INGESTION -------------
# Extract metadata 18-10-2025
def extract_layer_metadata(filepath): 
    """Ëxtract metadata from vector/raster geospatial datasets"""
    # sanitize and normalize the incoming filepath (users sometimes paste Python raw literals like r"C:\..")
    raw = str(filepath).strip()
    # strip surrounding quotes if present
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]
    # remove leading r or R if user copied a python raw string literal (e.g. r"C:\path")
    if len(raw) > 0 and raw[0] in ('r', 'R'):
        # if it looks like rC:\ or r"C:\ then drop the leading r/R
        raw = raw[1:]

    raw = os.path.expanduser(raw)
    raw = os.path.normpath(raw)

    metadata = {"path": raw, "type": None}

    if not os.path.exists(raw):
        metadata['error'] = f"Path does not exist: {raw}"
        return metadata

    ext = os.path.splitext(raw)[1].lower()

    try:
        if ext in ['.shp', '.geojson']:
            gdf = gpd.read_file(raw)
            metadata.update({
                'type': 'vector', 
                'crs': str(gdf.crs),
                'geometry_type': list(gdf.geom_type.unique()) if hasattr(gdf, 'geom_type') else [],
                'num_features': len(gdf), 
                'fields': list(gdf.columns), 
                'bounds': gdf.total_bounds.tolist() if hasattr(gdf, 'total_bounds') else []
            })
        ## Adjusting code to account for .gdb layers 10-11-2025
        elif raw.lower().endswith('.gdb') and os.path.isdir(raw):
            layers = fiona.listlayers(raw)
            print(f"[INFO] Found {len(layers)} layers in Geodatabase {raw}")

            for layer in layers:
                gdf = gpd.read_file(raw, layer=layer)
                layer_metadata = {
                    'type': 'vector', 
                    'layer_name': layer,
                    'crs': str(gdf.crs),
                    'geometry_type': list(gdf.geom_type.unique()) if hasattr(gdf, 'geom_type') else [],
                    'num_features': len(gdf), 
                    'fields': list(gdf.columns), 
                    'bounds': gdf.total_bounds.tolist() if hasattr(gdf, 'total_bounds') else []
                }
                metadata[f'layer_{layer}'] = layer_metadata
            metadata['type'] = 'geodatabase'     
            metadata['layers']  = layers
        ## Adjusting code to account for .gpkg layers 25-11-2025
        elif ext == ".gpkg":
            try:
                layers = fiona.listlayers(raw)
                metadata["type"] = "geopackage"
                metadata["layers"] = layers
                metadata["num_layers"] = len(layers)

                for layer in layers:
                    gdf = gpd.read_file(raw, layer=layer)
                    layer_meta = {
                        'type': 'vector', 
                        'layer_name': layer, 
                        'crs': str(gdf.crs), 
                        'geometry_type': list(gdf.geom_type.unique()) if hasattr(gdf, 'geom_type') else [],
                        'num_features': len(gdf), 
                        'fields': list(gdf.columns), 
                        'bounds': gdf.total_bounds.tolist() if hasattr(gdf, 'total_bounds') else []
                    }
                    metadata[f"layer_{layer}"] = layer_meta
            except Exception as e:
                metadata['error'] = f"Error reading GeoPackage: {str(e)}"


        elif ext in ['.tif', '.tiff']:
            with rasterio.open(raw) as src:
                metadata.update({
                    'type': 'raster', 
                    'crs': str(src.crs),
                    'width': src.width, 
                    'height': src.height, 
                    'count': src.count, 
                    'driver': src.driver, 
                    'dtype': src.dtypes[0] if src.dtypes else None, 
                    'bounds': list(src.bounds), 
                    'transform': str(src.transform)
                })
        else:
            metadata['error'] = f"Unsupported file extension: {ext}"
    except Exception as e:
        metadata['error'] = str(e)
    return metadata

#---------------------- METADATA ENGICHMENT WITH LLM -------------
def enrich_metadata(metadata_dict):
    """Uses Llama to generate semantic summary and keywords for each input layer"""
    llm = OllamaLLM(model="llama3.2", temperature=0)
    prompt = f"""
You are a geospatial metadata assistant.
Given the dataset metadata, generate a short semantic description including:
- Purpose or theme of the dataset
- Potential use cases (e.g., land use classification, transport, hydrology)
- Suggested keywords
Input metadata:
{json.dumps(metadata_dict, indent=2)}
Respond ONLY with a JSON object in this format:
{{
"description": "Provide a concise description here.", 
"theme": "Provide the main theme here.", 
"keywords": ["Provide", "relevant", "keywords"]
}}"""
    response = llm.invoke(prompt)
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        print("[WARN] Could not parse model response, skipping metadata enrichment")
        return None 
    try:
        data = json.loads(match.group(0))
        return data
    except json.JSONDecodeError:
        print("[WARN] Invalid JSON format in model output")
        return None 
    
############ Ingest metadata (to be continued)
# def store_input_layer_metadat(filepaths, persist_dir):
#     client = PersistentClient(path=persist_dir)
#     coll = client.get_or_create_collection("input_layers")

#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     docs, metas, ids = [], [], []
#     for fp in filepaths:
#         meta = extract_layer_metadata(fp)
#         meta_text = json.dumps(meta, indent=2)
#         docs.append(meta_text)
#         metas.append(meta)
#         ids.append(os.path.basename(fp))
    
#     embeddings = model.encode(docs, convert_to_numpy=True).to_list()
#     coll.add(documents=docs, metadatas=metas, ids=ids, embeddings=embeddings)
#     print(f"[INFO] Added {len(filepaths)} layers to input_layers collection.")

##### RETRIEVAL OF INPUT LAYERS (TO BE CONTINUED)
## 11-11-2025 The retrieval function has been adjusted now to include also an 
# enrichment component that uses an LLM to create semantic summaries

def ingest_input_layers(persist_dir, layer_paths):
    """Extracts metadata from user input data and stores them in a separate
    collection 'input_layers' in ChromaDB"""
    os.makedirs(persist_dir, exist_ok=True)
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    client = PersistentClient(path=persist_dir)
    coll = client.get_or_create_collection("input_layers")

    docs, metas, ids = [], [], []
    
    for fp in layer_paths:
        meta = extract_layer_metadata(fp)
        ## Adjusted to handle geodatabase layers 10-11-2025
        if meta.get('error'):
            print(f"[WARN] Skipping {fp} due to error: {meta['error']}")
        if meta.get('type') == 'geodatabase':
            for k, layer_meta in meta.items():
                if not k.startswith('layer_'):
                    continue 
                # print(f"[INFO] Enriching metadata for layer: {layer_meta.get('layer_name', 'unknown')}")
                # enriched = enrich_metadata(layer_meta) -> DO NOT INCLUDE FOR BASELINE MODEL
                # if enriched:
                #     layer_meta.update(enriched)
                layer_text = json.dumps(layer_meta, indent=2)
                docs.append(layer_text)
                metas.append(layer_meta)
                ids.append(make_id("input", os.path.basename(fp), k))
        else:
            # print(f"[INFO] Enriching metadata for: {os.path.basename(fp)}")
            # enriched = enrich_metadata(meta)
            # if enriched:
            #     meta.update(enriched)
            meta_text = json.dumps(meta, indent=2)
            docs.append(meta_text)
            metas.append(meta)
            # Check if input is valid here 
            ids.append(make_id("input_layer", os.path.basename(fp)))

    def sanitize_metadata(meta):
        clean_meta = {}
        for k, v in (meta or {}).items():
            if v is None:
                clean_meta[k] = "" # Replacing None with empty string
            elif isinstance(v, (str, int, float, bool)) or v is None:
                clean_meta[k] = v
            elif isinstance(v, (list, dict)):
                clean_meta[k] = json.dumps(v, ensure_ascii=False)
            else:
                clean_meta[k] =str(v)
        return clean_meta
    metas = [sanitize_metadata(m) for m in metas if m is not None]
    # embeddings = model.encode(docs,convert_to_numpy=True).tolist()
    embeddings = embedder.embed_documents(docs)
    coll.add(documents=docs, metadatas=metas, ids=ids, embeddings=embeddings)

    print(f"[INFO] Added {len(layer_paths)} layer(s) to collection 'input_layers'.")

# --------------------- VALIDATION ---------------------
def validate_query(version, persist_dir, query, top_k=3):
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
    client = chromadb.Client(settings)
    coll = client.get_collection(f"pyqgis_{version.replace('.', '_')}_docs")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = embedder.encode([query], convert_to_numpy=True).tolist()
    res = coll.query(query_embeddings=q_emb, n_results=top_k, include=["documents", "metadatas"])

    for i, (doc, md) in enumerate(zip(res["documents"][0], res["metadatas"][0])):
        print(f"\nResult {i+1}:")
        print(f"URL: {md['url']}")
        print(f"Heading: {md['heading']}")
        print(f"Module: {md['module']}")
        print(doc[:400], "...")
    return res

# --------------------- MAIN ---------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=['pyqgis', 'python', 'both'], required=True, 
                   help="Which ingestion mode to run?:pyqgis, python or both")
    # p.add_argument("--version", required=True)
    p.add_argument("--version", default="3.34", help="QGIS version for PyQGIS docs")
    p.add_argument("--persist_dir", default="./chroma_db")
    p.add_argument("--prompt_examples", default=None)
    p.add_argument("--validate_query", default=None)
    p.add_argument("--input_layers", nargs="*", default=None, help="List of geospatial layer file paths (e.g. shapefile, GeoTIFF) to ingest as metadata")
    args = p.parse_args()

    if args.mode == 'pyqgis':
        ingest(args.version, args.persist_dir, args.prompt_examples)

    elif args.mode == 'python':
        ingest_python_geo_docs(args.persist_dir)
    
    elif args.mode == 'both':
        ingest(args.version, args.persist_dir, args.prompt_examples)
        ingest_python_geo_docs(args.persist_dir)
    
    elif args.mode == 'none':
        print("[INFO] No ingestion selected. Skipping data load")

    if args.input_layers:
        ingest_input_layers(args.persist_dir,args.input_layers)

    # ingest(args.version, args.persist_dir, args.prompt_examples)
    if args.validate_query:
        validate_query(args.version, args.persist_dir, args.validate_query)
