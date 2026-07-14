import chromadb
from typing import List
import os

class LongTermMemory:
    """
    Manages long-term user facts using ChromaDB vector store.
    Stores user preferences and recalls them based on semantic similarity.
    """
    def __init__(self):
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        
        self.client = chromadb.HttpClient(host=host, port=port)
        # Create a specific collection for user profiles/facts
        self.collection = self.client.get_or_create_collection(name="user_facts")

    def save_fact(self, user_id: str, fact: str):
        """Saves a new fact about the user to the vector database."""
        # Create a unique ID for the fact to avoid exact duplicates
        fact_id = f"{user_id}_{hash(fact)}"
        
        self.collection.upsert(
            documents=[fact],
            metadatas=[{"user_id": user_id}],
            ids=[fact_id]
        )
        print(f"[Memory] Saved new fact for {user_id}: {fact}")

    def retrieve_relevant_facts(self, user_id: str, query: str, n_results: int = 3) -> List[str]:
        """Retrieves facts relevant to the current conversation from the DB."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"user_id": user_id}
            )
            
            if results["documents"] and len(results["documents"]) > 0:
                return results["documents"][0]
            return []
        except Exception as e:
            print(f"[Memory] Retrieval error: {e}")
            return []