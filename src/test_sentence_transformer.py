from sentence_transformers import SentenceTransformer

print("Loading SentenceTransformer...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer loaded successfully.")
except Exception as e:
    print(f"Error: {str(e)}")