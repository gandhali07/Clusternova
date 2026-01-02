from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
import uvicorn

# ----------------------------
# Initialize API
# ----------------------------
app = FastAPI(title="Academic Paper Search API v2")

# Allow requests from React or local HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to localhost:3000 if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load dataset
# ----------------------------
file_path = r"D:\Downloads\arXiv-DataFrame.csv\arXiv-DataFrame.csv"
df = pd.read_csv(file_path)
df.fillna("", inplace=True)
df["text"] = df["Title"] + " " + df["Summary"]

# ----------------------------
# Vectorize
# ----------------------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["text"])

# ----------------------------
# Clustering
# ----------------------------
kmeans = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=100)
df["cluster"] = kmeans.fit_predict(tfidf_matrix)

# ----------------------------
# Pydantic model
# ----------------------------
class Paper(BaseModel):
    title: str
    author: str
    summary: str
    link: str
    cluster: int
    similarity: float

# ----------------------------
# Search Endpoint
# ----------------------------
@app.get("/search")
def search_papers(query: str = Query(..., min_length=2), top_k: int = 5, cluster: int | None = None):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    if cluster is not None:
        mask = df["cluster"] == cluster
        similarities = similarities * mask  # zero out other clusters

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for i in top_indices:
        if similarities[i] > 0:
            row = df.iloc[i]
            results.append({
                "title": str(row["Title"]),
                "author": str(row["Author"]),
                "summary": str(row["Summary"]),
                "link": str(row["Link"]),
                "cluster": int(row["cluster"]),
                "similarity": float(similarities[i])
            })

    return results  # FastAPI automatically serializes JSON
            

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7874)
