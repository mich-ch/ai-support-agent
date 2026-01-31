import sys
import os
# Fix paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracing import setup_tracing
from src.data import initialize_faq_embeddings
from src.agent import handle_support_query

if __name__ == "__main__":
    setup_tracing()
    initialize_faq_embeddings() # Πρέπει να τρέξει μια φορά για να γεμίσει τα vectors
    
    print("\n=== Support Bot CLI ===")
    
    # Test Queries
    queries = [
        "Where is my order ORD-12345?",
        "How do I reset my password?",
        "What is the status of ORD-99999?" # Λάθος κωδικός
    ]
    
    for q in queries:
        print("-" * 50)
        response = handle_support_query(q)
        print(f"📤 Response: {response}")
        
    print("\n✅ Check traces at http://localhost:6006")