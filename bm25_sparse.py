from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


import requests
from langchain_core.runnables import Runnable

class OllamaRunnable(Runnable):
    def __init__(self, model="gemma3:4b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def invoke(self, input, config=None):

        # 🔥 FIX: convert ChatPromptValue → string
        if hasattr(input, "to_string"):
            prompt = input.to_string()
        elif hasattr(input, "messages"):
            prompt = "\n".join([m.content for m in input.messages])
        else:
            prompt = str(input)

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json().get("response", "")

llm = OllamaRunnable(model="gemma3:4b")

chunks = [
    "Iran, officially the Islamic Republic of Iran...",
    "It borders Iraq to the west...",
    "With a population of over 92 million...",
    "It is divided into five regions...",
    "Tehran is the nation's capital..."
]

# Convert to Document objects
documents = [Document(page_content=c) for c in chunks]

# BM25 retriever (IMPORTANT: k=5)
retriever = BM25Retriever.from_documents(documents, k=5)

# Prompt (from exercise or similar)
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context.

Context:
{context}

Question:
{question}
""")

# LCEL chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run
print(chain.invoke("Where is the largest city of Iran?"))