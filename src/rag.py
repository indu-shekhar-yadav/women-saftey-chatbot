import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

class RAG:
    def __init__(self, data_dir='../data'):
        # Resolve the absolute path to the data directory
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of rag.py (src/)
        data_dir = os.path.join(base_dir, data_dir)  # Resolve relative path
        print(f"Loading RAG with resolved data_dir: {data_dir}")
        
        try:
            print("Initializing SentenceTransformer...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer initialized.")
        except Exception as e:
            print(f"Error initializing SentenceTransformer: {str(e)}")
            raise

        try:
            print("Creating FAISS index...")
            self.index = faiss.IndexFlatL2(384)
            print("FAISS index created.")
        except Exception as e:
            print(f"Error creating FAISS index: {str(e)}")
            raise

        self.responses = []
        self.load_data(data_dir)

    def load_data(self, data_dir):
        files = ['legal_rights.json', 'safety_tips.json', 'self_defense.json']
        queries = []
        for file in files:
            file_path = f"{data_dir}/{file}"
            print(f"Attempting to load: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        queries.append(item['query'])
                        self.responses.append(item['response'])
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                raise
        embeddings = self.encoder.encode(queries)
        self.index.add(np.array(embeddings))

    def retrieve(self, query):
        query_embedding = self.encoder.encode([query])
        _, indices = self.index.search(np.array(query_embedding), 1)
        return self.responses[indices[0][0]]

if __name__ == "__main__":
    rag = RAG()
    response = rag.retrieve("How to stay safe at night?")
    print(f"Response: {response}")