"""
vectorstore_fa.py — Vector Store with Persian documents
========================================================
Same concepts as vectorstore.py but with:
  - Persian documents
  - Multilingual embedding model (supports 50+ languages including Persian)
  - Persian queries
"""

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

AVALAI_BASE_URL = "https://api.avalai.ir/v1"
AVALAI_API_KEY = os.getenv("AVALAI_API_KEY")

# 1. Multilingual Embedding Model

# all-MiniLM-L6-v2     → English only ❌
# paraphrase-multilingual-MiniLM-L12-v2 → 50+ languages including Persian ✅

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 2. Persian Documents

docs = [
    Document(page_content="پایتون یک زبان برنامه‌نویسی محبوب برای هوش مصنوعی است."),
    Document(page_content="لنگ‌چین یک فریمورک برای ساخت اپلیکیشن‌های مبتنی بر مدل‌های زبانی است."),
    Document(page_content="RAG مخفف Retrieval Augmented Generation است."),
    Document(page_content="FAISS یک کتابخانه متن‌باز از فیسبوک برای جستجوی سریع وکتور است."),
    Document(page_content="Chroma یک پایگاه داده وکتور متن‌باز است که ذخیره‌سازی دائمی دارد."),
    Document(page_content="Embedding یعنی تبدیل متن به یک آرایه عددی با معنا."),
    Document(page_content="مدل‌های زبانی بزرگ مثل GPT-4 می‌توانند متن تولید کنند."),
    Document(page_content="در RAG ابتدا اطلاعات مرتبط بازیابی می‌شود، سپس مدل جواب می‌دهد."),
]

# 3. FAISS with Persian documents

faiss_db = FAISS.from_documents(docs, embeddings)

query = "چطور متن را به عدد تبدیل کنم؟"
results = faiss_db.similarity_search(query, k=2)

print("=== FAISS: similarity_search (Persian) ===")
for r in results:
    print("-", r.page_content)

results_with_score = faiss_db.similarity_search_with_score(query, k=2)
print("\n=== FAISS: with score ===")
for doc, score in results_with_score:
    print(f"score={score:.4f} | {doc.page_content}")

# 4. Chroma with Persian documents

chroma_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db_fa",
    collection_name="persian_rag_collection"
)

results_chroma = chroma_db.similarity_search("بازیابی اطلاعات در RAG چطور کار می‌کند؟", k=2)
print("\n=== Chroma: similarity_search (Persian) ===")
for r in results_chroma:
    print("-", r.page_content)

# 5. Retriever

retriever = faiss_db.as_retriever(search_kwargs={"k": 3})

retriever_mmr = faiss_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)

retrieved = retriever.invoke("پایگاه داده وکتور چیست؟")
print("\n=== Retriever results (Persian) ===")
for doc in retrieved:
    print("-", doc.page_content)

# 6. Chain with Persian prompt

llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=AVALAI_BASE_URL,
    api_key=AVALAI_API_KEY
)

prompt = ChatPromptTemplate.from_template("""
با توجه به اطلاعات زیر به سوال پاسخ بده. پاسخ را به فارسی بنویس.

اطلاعات:
{context}

سوال: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Uncomment to run (requires Avalai API key):
# response = chain.invoke("RAG چیست و چطور کار می‌کند؟")
# print("\n Chain Response")
# print(response)

print("\n All sections ready. Uncomment chain.invoke and run!")