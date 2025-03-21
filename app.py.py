
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = FastAPI()

# ✅ Load Data
df = pd.read_csv("cleaned_hotel_bookings.csv")
df['text_data'] = df.apply(lambda row: f"Hotel: {row['hotel']}, Date: {row['reservation_status_date']}, ADR: {row['adr']}, Country: {row['country']}", axis=1)

# ✅ Load Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Load Precomputed Embeddings
embeddings = np.load("precomputed_embeddings.npy")

# ✅ Load FAISS Index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# ✅ Load LLM (GPT-2 for Speed)
qa_model = pipeline("text-generation", model="gpt2", max_new_tokens=50)

class QueryModel(BaseModel):
    question: str

def search_booking(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [df.iloc[idx]['text_data'] for idx in indices[0]]

def ask_llm(query):
    retrieved_context = " ".join(search_booking(query))
    prompt = f"Context: {retrieved_context} \nQuestion: {query} \nAnswer:"
    
    response = qa_model(prompt, max_new_tokens=50)
    return response[0]['generated_text']

@app.get("/analytics")
def get_analytics():
    total_revenue = df['adr'].sum()
    cancellation_rate = (df['is_canceled'].sum() / len(df)) * 100
    return {"total_revenue": total_revenue, "cancellation_rate": f"{cancellation_rate:.2f}%"}

@app.post("/ask")
def ask(query: QueryModel):
    answer = ask_llm(query.question)
    return {"answer": answer}
