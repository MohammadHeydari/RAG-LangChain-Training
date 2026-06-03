from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


chunks = [
    "Iran, officially the Islamic Republic of Iran, historically known as Persia, is a country in West Asia.",
    "It borders Iraq to the west, Turkey, Azerbaijan, and Armenia to the northwest, the Caspian Sea to the north, Turkmenistan to the northeast, Afghanistan to the east, Pakistan to the southeast, and the Gulf of Oman and the Persian Gulf to the south.",
    "With a population of over 92 million, Iran ranks 17th globally in both geographic size and population.",
    "It is divided into five regions with 31 provinces.",
    "Tehran is the nation's capital and largest city and serves as its primary economic centre."

]

documents = [Document(page_content=c) for c in chunks]

retriever = BM25Retriever.from_documents(documents, k=3)

query = "Where is the Largest City of Iran?"
results = retriever.invoke(query)

print("Top results:")
for i, doc in enumerate(results):
    print(f"Result {i+1}:\n{doc.page_content}\n{'-'*40}")