from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("data/Designing_Data_Intensive_Applications_The_Big_Ideas_Behind_Reliable.pdf")
document = loader.load()


embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)


semantic_splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="gradient",
    breakpoint_threshold_amount=0.8
)


chunks = semantic_splitter.split_documents(document)


print(chunks[0])