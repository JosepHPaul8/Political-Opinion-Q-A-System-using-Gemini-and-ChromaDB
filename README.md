# Political-Opinion-Q-A-System-using-Gemini-and-ChromaDB
A RAG-based question answering system built with LangChain, Gemini, and ChromaDB to analyze public opinion from political tweets. The system retrieves semantically relevant tweets with sentiment-aware filtering and generates reliable, context-grounded responses.
# RAG-based Political Opinion Analysis using Twitter Data and Gemini

This project implements a **Retrieval-Augmented Generation (RAG)** system that analyzes public political opinions using Twitter data.  
Users can ask open-ended questions about a politician, and the system generates **grounded answers strictly based on real tweets**, not external internet knowledge.

---

## 🚀 Project Overview

Social media platforms like Twitter contain large volumes of public opinion data.  
This project leverages **semantic search and generative AI** to extract meaningful insights from political tweets.

Instead of simple sentiment classification, the system supports **analytical question answering**, such as:
- What are people criticizing the most?
- What do supporters appreciate?
- What themes appear across public opinion?

---

## 🧠 How It Works (RAG Pipeline)

1. Tweets are loaded from a CSV file  
2. Each tweet is converted into a vector embedding using **Gemini embeddings**  
3. Embeddings are stored in **ChromaDB** (vector database) with sentiment metadata  
4. User questions are converted into embeddings  
5. Relevant tweets are retrieved using semantic similarity  
6. **Gemini LLM** generates answers grounded in retrieved tweets  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Framework:** LangChain  
- **LLM:** Google Gemini (`gemini-pro`)  
- **Embeddings:** Gemini Embeddings (`embedding-001`)  
- **Vector Database:** ChromaDB  
- **Data Source:** Twitter CSV dataset  
- **Environment:** Conda / Python 3.10+  

---

## 📂 Project Structure

