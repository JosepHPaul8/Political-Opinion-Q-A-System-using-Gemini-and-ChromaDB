import pandas as pd
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load CSV
df = pd.read_csv("data/Twitter_Data.csv")
df = df.dropna(subset=["clean_text"])

documents = []

for _, row in df.iterrows():
    documents.append(
        Document(
            page_content=row["clean_text"],
            metadata={"sentiment": (row["category"])}
        )
    )

# Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004"
)

# Create Chroma DB (persistent)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="chroma_db"
)

vectorstore.persist()

print("✅ Tweets successfully stored in ChromaDB")
