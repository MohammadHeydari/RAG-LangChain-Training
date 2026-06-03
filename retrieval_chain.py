from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

docs = [
    "Mohammad Heydari is an AI Data Engineer at SiCS Company",
    "SiCS Company works in AI data engineering solutions",
    "FAISS is used for similarity search in vector databases"
]


vector_store = FAISS.from_texts(docs, embeddings)


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

prompt = """
Answer only using the context below.

Context:
{context}

Question:
{question}
"""

prompt_template = ChatPromptTemplate.from_template(prompt)

def format_docs(docs):
    return "\n".join([d.page_content for d in docs])

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

print(chain.invoke("What is SiCS?"))