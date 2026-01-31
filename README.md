# 🤖 AI Customer Support Agent with Semantic Routing & RAG

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green)
![Arize Phoenix](https://img.shields.io/badge/Observability-Phoenix-orange)

## 📋 Overview

This project implements an intelligent **Customer Support Agent** capable of handling different types of user queries dynamically. Unlike simple chatbots, this agent uses a **Semantic Router** architecture to classify user intent and route the request to the most appropriate subsystem:

1.  **Transactional Queries (Order Status):** Handled via **Function Calling (Tools)** to simulate database lookups.
2.  **Informational Queries (FAQs):** Handled via **RAG (Retrieval Augmented Generation)** using vector embeddings.

The entire execution flow is traced and observed using **Arize Phoenix**, allowing for detailed debugging and performance monitoring.

## 🏗️ Architecture

The system follows a decision-based workflow:

```mermaid
graph TD
    A[👤 User Query] -->|Input| B{🧠 Semantic Router}
    
    B -->|Intent: Order Status| C[🔧 Tool Execution Branch]
    B -->|Intent: General FAQ| D[📚 RAG Branch]
    
    C -->|Extract Order ID| E[Function Call: lookupOrderStatus]
    E -->|Query Mock DB| F[(Order Database)]
    F -->|Return Data| G[Generate Response]
    
    D -->|Embed Query| H[Vector Search]
    H -->|Retrieve Context| I[(FAQ Knowledge Base)]
    I -->|Augment Prompt| G
    
    G --> J[📝 Final Answer]