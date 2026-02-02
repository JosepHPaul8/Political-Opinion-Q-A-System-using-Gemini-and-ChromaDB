from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_chroma import Chroma

# Load embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# Load Chroma DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Detect sentiment intent
def detect_sentiment_filter(query: str):
    q = query.lower()
    if any(word in q for word in ["critic", "negative", "complaint"]):
        return {"sentiment": -1}
    if any(word in q for word in ["prais", "positive", "support"]):
        return {"sentiment": 1}
    return {} # Return empty dict instead of None

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-robotics-er-1.5-preview", # Updated to a valid model name
    temperature=0.7
)

while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        break

    sentiment_filter = detect_sentiment_filter(query)
    
    # Only apply filter if it's not empty
    search_args = {"k": 10}
    if sentiment_filter:
        search_args["filter"] = sentiment_filter

    retriever = vectorstore.as_retriever(
        search_kwargs=search_args
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # Use .invoke() instead of .run() as .run() is being phased out
    response = qa.invoke({"query": query})
    print("\nAnswer:\n", response["result"])