from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME
from pprint import pprint
import numpy as np

class EmbeddingService:

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
        )
        print("Embedding model loaded.")

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text)

        return embedding.tolist()

    def embed_documents(
        self,
        documents: list[str],
        ) -> list[list[float]]:
            return self.model.encode(documents).tolist()

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# if __name__ == "__main__":
#     embedding_service = EmbeddingService()
#     sample_text1 = "Employees receive 12 casual leaves per year."
#     sample_text2 = "Workers are entitled to 12 leaves." 
#     sample_text3 = "Espaneol won the world cup this year."

#     embedding1 = embedding_service.embed_text(sample_text1)
#     embedding2 = embedding_service.embed_text(sample_text2)
#     embedding3 = embedding_service.embed_text(sample_text3)

    
    # print(f"Embedding length of 1: {len(embedding1)}")  # Should print the length of the embedding vector
    # pprint(embedding1[:10])  # Print the first 10 dimensions of the embedding
    
    # print(f"Embedding length of 2: {len(embedding2)}")  # Should print the length of the embedding vector
    # pprint(embedding2[:10])  # Print the first 10 dimensions of the embedding
   
    # print(f"Embedding length of 3 : {len(embedding3)}")  # Should print the length of the embedding vector
    # pprint(embedding3[:10])  # Print the first 10 dimensions of the embedding

    # print(
    #         cosine_similarity(
    #     embedding1,
    #     embedding2
    #     )
    # )

    # print(
    #  cosine_similarity(
    #     embedding1,
    #     embedding3
    #     )
    # )

    # print(
    #     cosine_similarity(
    #     embedding2,
    #     embedding3
    #     )
    # )
