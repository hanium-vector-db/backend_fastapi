from langchain.schema import Document
from langchain.vectorstores import Chroma

class RetrievalService:
    def __init__(self, vector_db_path: str):
        self.vector_db = Chroma(persist_directory=vector_db_path)

    def retrieve_documents(self, query: str, k: int = 2) -> list[Document]:
        """Retrieve relevant documents based on the query."""
        results = self.vector_db.similarity_search(query, k=k)
        return results

    def add_document(self, document: Document):
        """Add a new document to the vector database."""
        self.vector_db.add_documents([document])