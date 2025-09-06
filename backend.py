import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# ------------------------
# Load Data
# ------------------------
df = pd.read_csv("profiles_sample_100.csv")

def extract_years(exp_text):
    m = re.search(r"\d+", str(exp_text) if pd.notnull(exp_text) else "")
    return int(m.group(0)) if m else 0

df["experience_years"] = df.get("experience", "").apply(extract_years)
df["rating"] = df.get("rating", df.get("ratings", 0)).fillna(0).astype(int)
df["full_text"] = df.apply(
    lambda r: f"{r['name']} - {r['skills']} - {r['description']} - {r['experience']} - {r['location']}", axis=1
)

# ------------------------
# Embeddings
# ------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

if "embedding" not in df.columns:
    df["embedding"] = model.encode(df["full_text"].tolist(), batch_size=32, show_progress_bar=True).tolist()
    df.to_parquet("profiles_with_embeddings.parquet")

# ------------------------
# Persistent Chroma
# ------------------------
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="profiles", metadata={"hnsw:space": "cosine"})

if collection.count() == 0:
    collection.add(
        documents=df["full_text"].tolist(),
        embeddings=list(df["embedding"]),
        ids=df.index.astype(str).tolist(),
        metadatas=df[["name", "skills", "description", "experience", "experience_years", "location", "rating"]].to_dict("records")
    )

# ------------------------
# Gemini setup via LangChain
# ------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),   # ✅ fixed: read from environment
    temperature=0.2
)

prompt = PromptTemplate(
    input_variables=["query", "profiles"],
    template="""
    You are an AI assistant that re-ranks candidate profiles for hiring.

    User query: {query}

    Candidate Profiles:
    {profiles}

    Task:
    - Rank only by best match.
    - Respond with ONLY the names in ranked order, comma-separated (no extra text).
    """
)
rerank_chain = LLMChain(llm=llm, prompt=prompt)

# ------------------------
# Search function
# ------------------------
def search_profiles(query: str, top_k: int = 5, min_rating: int | None = None, min_exp: int | None = None):
    query_emb = model.encode(query).tolist()

    filters = []
    if min_rating is not None:
        filters.append({"rating": {"$eq": min_rating}})
    if min_exp is not None:
        filters.append({"experience_years": {"$eq": min_exp}})

    where = {"$and": filters} if len(filters) > 1 else (filters[0] if filters else None)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=max(top_k * 5, top_k),
        where=where
    )

    profiles = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        similarity = 1 - dist
        boosted = similarity + (0.3 * (int(meta.get("rating", 0)) / 5)) + (0.15 * (int(meta.get("experience_years", 0)) / 10))
        profiles.append((boosted, meta))

    if not profiles:
        return []

    # Prepare text for reranking
    profiles_text = "\n".join([f"{p[1]['name']} - {p[1]['skills']} - {p[1]['description']}" for p in profiles])

    try:
        reranked = rerank_chain.run(query=query, profiles=profiles_text)
        reranked_names = [name.strip() for name in reranked.split(",") if name.strip()]
        ranked_profiles = [p for name in reranked_names for p in profiles if p[1]["name"] == name]
        if ranked_profiles:
            return [(score, meta) for score, meta in ranked_profiles[:top_k]]
    except Exception as e:
        print("⚠️ Gemini reranking failed, fallback to vector search:", e)

    return [(score, meta) for score, meta in sorted(profiles, key=lambda x: x[0], reverse=True)[:top_k]]
