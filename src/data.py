import numpy as np
from typing import Dict, List, TypedDict, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# --- Τύποι Δεδομένων ---
class FAQEntry(TypedDict):
    id: int
    question: str
    answer: str
    category: str
    embedding: Optional[List[float]]

# --- Βάση Παραγγελιών ---
ORDER_DATABASE: Dict[str, Dict[str, str]] = {
    "ORD-12345": {"status": "shipped", "carrier": "FedEx", "trackingNumber": "1234567890", "eta": "December 11, 2025"},
    "ORD-67890": {"status": "processing", "carrier": "pending", "trackingNumber": "pending", "eta": "December 15, 2025"},
    "ORD-11111": {"status": "delivered", "carrier": "UPS", "trackingNumber": "9876543210", "eta": "Delivered December 5, 2025"},
}

# --- Βάση FAQs ---
FAQ_DATABASE: List[FAQEntry] = [
    {"id": 1, "question": "How do I reset my password?", "answer": "Go to Settings > Security > Reset Password.", "category": "Account", "embedding": None},
    {"id": 2, "question": "What's your refund policy?", "answer": "We offer full refunds within 30 days of purchase.", "category": "Billing", "embedding": None},
    {"id": 3, "question": "How do I cancel my subscription?", "answer": "Go to Account Settings > Subscription > Cancel.", "category": "Billing", "embedding": None},
    {"id": 4, "question": "What payment methods do you accept?", "answer": "We accept Visa, Mastercard, PayPal.", "category": "Billing", "embedding": None},
    {"id": 5, "question": "How do I update my profile?", "answer": "Go to Account Settings > Profile.", "category": "Account", "embedding": None},
]

# --- Helper Functions ---
def cosine_similarity(a: List[float], b: List[float]) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def initialize_faq_embeddings():
    """Δημιουργεί διανύσματα (embeddings) για τα FAQs μια φορά στην αρχή."""
    print("📚 Initializing FAQ embeddings...")
    for faq in FAQ_DATABASE:
        if faq["embedding"] is None:
            response = client.embeddings.create(model="text-embedding-ada-002", input=faq["question"])
            faq["embedding"] = response.data[0].embedding
    print("✅ FAQ embeddings initialized")