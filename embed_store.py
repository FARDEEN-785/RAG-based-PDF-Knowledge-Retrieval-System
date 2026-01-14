import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def build(self, chunks: list[str]):
        self.chunks = chunks
        embeddings = self.model.encode(chunks, convert_to_numpy=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, question: str, top_k=3, max_distance=1.2):
        query_vec = self.model.encode([question], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for d, i in zip(distances[0], indices[0]):
            if d < max_distance:
                results.append(self.chunks[i])

        return results
