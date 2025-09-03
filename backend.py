# import re
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import chromadb

# # ------------------------
# # Load Data
# # ------------------------
# df = pd.read_csv("profiles_sample_100.csv")

# def extract_years(exp_text):
#     # grab the first integer in the text; default 0 if none
#     m = re.search(r"\d+", str(exp_text) if pd.notnull(exp_text) else "")
#     return int(m.group(0)) if m else 0

# df["experience_years"] = df.get("experience", "").apply(extract_years)

# # Ensure rating column exists & is int
# if "rating" not in df.columns:
#     df["rating"] = 0
# df["rating"] = df["rating"].fillna(0).astype(int)

# # ------------------------
# # Chroma & Embeddings
# # ------------------------
# model = SentenceTransformer("all-MiniLM-L6-v2")
# chroma_client = chromadb.Client()

# # Recreate collection each run to avoid stale metadata
# existing = [c.name for c in chroma_client.list_collections()]
# if "profiles" in existing:
#     chroma_client.delete_collection("profiles")
# collection = chroma_client.create_collection(name="profiles", metadata={"hnsw:space": "cosine"})

# # Index profiles
# for idx, row in df.iterrows():
#     text = f"{row['name']} - {row['skills']} - {row['description']} - {row['experience']} - {row['location']}"
#     embedding = model.encode(text).tolist()
#     collection.add(
#         documents=[text],
#         embeddings=[embedding],
#         ids=[str(idx)],
#         metadatas=[{
#             "name": row.get("name",""),
#             "skills": row.get("skills",""),
#             "description": row.get("description",""),
#             "experience": row.get("experience",""),
#             "experience_years": int(row.get("experience_years", 0)),
#             "location": row.get("location",""),
#             "rating": int(row.get("rating", 0)),
#         }]
#     )

# # ------------------------
# # Hybrid Search
# # ------------------------
# def search_profiles(
#     query: str,
#     top_k: int = 5,
#     rating_weight: float = 0.3,
#     exp_weight: float = 0.15,
#     min_rating: int | None = None,   # None = no filter
#     min_exp: int | None = None       # None = no filter
# ):
#     query_embedding = model.encode(query).tolist()
#     results = collection.query(query_embeddings=[query_embedding], n_results=max(top_k * 5, top_k))

#     profiles = []
#     for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
#         similarity = 1 - dist
#         rating = int(meta.get("rating", 0))
#         exp = int(meta.get("experience_years", 0))

#         # --- Apply exact match filters ---
#         if (min_rating is not None and rating != min_rating):
#             continue
#         if (min_exp is not None and exp != min_exp):
#             continue

#         boosted = similarity + (rating_weight * rating) + (exp_weight * exp)
#         profiles.append((boosted, meta))

#     return sorted(profiles, key=lambda x: x[0], reverse=True)[:top_k]

import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# ------------------------
# Load Data
# ------------------------
df = pd.read_csv("profiles_sample_100.csv")

def extract_years(exp_text):
    # grab the first integer in the text; default 0 if none
    m = re.search(r"\d+", str(exp_text) if pd.notnull(exp_text) else "")
    return int(m.group(0)) if m else 0

df["experience_years"] = df.get("experience", "").apply(extract_years)

# ✅ Fix: standardize column name to "rating"
if "rating" not in df.columns:
    if "ratings" in df.columns:   # handle plural version
        df.rename(columns={"ratings": "rating"}, inplace=True)
    else:
        df["rating"] = 0

df["rating"] = df["rating"].fillna(0).astype(int)

# ------------------------
# Chroma & Embeddings
# ------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

# Recreate collection each run to avoid stale metadata
existing = [c.name for c in chroma_client.list_collections()]
if "profiles" in existing:
    chroma_client.delete_collection("profiles")
collection = chroma_client.create_collection(name="profiles", metadata={"hnsw:space": "cosine"})

# Index profiles
for idx, row in df.iterrows():
    text = f"{row['name']} - {row['skills']} - {row['description']} - {row['experience']} - {row['location']}"
    embedding = model.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(idx)],
        metadatas=[{
            "name": row.get("name",""),
            "skills": row.get("skills",""),
            "description": row.get("description",""),
            "experience": row.get("experience",""),
            "experience_years": int(row.get("experience_years", 0)),
            "location": row.get("location",""),
            "rating": int(row.get("rating", 0)),   # ✅ fixed
        }]
    )

# ------------------------
# Hybrid Search
# ------------------------
def search_profiles(
    query: str,
    top_k: int = 5,
    rating_weight: float = 0.3,
    exp_weight: float = 0.15,
    min_rating: int | None = None,   # None = no filter
    min_exp: int | None = None       # None = no filter
):
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=max(top_k * 5, top_k))

    profiles = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        similarity = 1 - dist
        rating = int(meta.get("rating", 0))
        exp = int(meta.get("experience_years", 0))

        # --- Apply exact match filters ---
        if (min_rating is not None and rating != min_rating):
            continue
        if (min_exp is not None and exp != min_exp):
            continue

        boosted = similarity + (rating_weight * rating) + (exp_weight * exp)
        profiles.append((boosted, meta))

    return sorted(profiles, key=lambda x: x[0], reverse=True)[:top_k]
