import chromadb

from config import(
    CHROMA_DB_PATH,
    COLLECTION_NAME,
)

class VectorDB : 

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

        self.collection = (
            self.client.get_or_create_collection(name=COLLECTION_NAME)
        )

    def add_documents(self , 
                      chunks: list[str] , 
                      embeddings:list[list[float]], 
                      source:str,
                      document_id:str,
                      page:int,
                      stored_filename: str):
        ids = []
        metadatas = []

        for i in range(len(chunks)):
            ids.append(f"{document_id}_page_{page}_chunk_{i}")
            metadatas.append({"source": source, 
                               "document_id": document_id,
                               "page": page , 
                               "chunk_index": i,
                               "stored_filename": stored_filename})

        self.collection.add(
            ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def reset_collection(self):

        self.client.delete_collection(COLLECTION_NAME)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def search(self ,
               query_embedding : list[float],
               top_k : int = 5):

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings = query_embedding,
            n_results = top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ],
        )
        matches = []

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, distance in zip(docs, metadatas, distances):
            matches.append({
            "document": doc,
            "metadata": meta,
            "distance": distance,
        })
        return matches 

    def list_documents(self) -> list[dict]:
        """Return one summary record for each indexed document."""
        records = {}
        result = self.collection.get(include=["metadatas"])

        for metadata in result["metadatas"]:
            document_id = metadata.get("document_id")
            if not document_id:
                continue

            record = records.setdefault(
                document_id,
                {
                    "pdf_id": document_id,
                    "source": metadata["source"],
                    "chunk_count": 0,
                },
            )
            record["chunk_count"] += 1

        return sorted(records.values(), key=lambda record: record["source"].lower())

    def delete_document(self, document_id: str) -> dict | None:
        """Delete every vector belonging to one document and return its metadata."""
        result = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )

        if not result["ids"]:
            return None

        metadata = result["metadatas"][0]
        self.collection.delete(ids=result["ids"])
        return metadata



# if __name__ == "__main__":
#     db =  vectorDB()

#     print("Vecotr database initialized successfully.")
#     print(f"Collection Name: {db.collection.name}")
