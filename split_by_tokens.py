import tiktoken
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("data/Designing_Data_Intensive_Applications_The_Big_Ideas_Behind_Reliable.pdf")
document = loader.load()


encoding = tiktoken.encoding_for_model("gpt-4o-mini")


token_splitter = TokenTextSplitter(
    encoding_name=encoding.name,
    chunk_size=100,
    chunk_overlap=10
)


chunks = token_splitter.split_documents(document)

for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:")
    print("Tokens:", len(encoding.encode(chunk.page_content)))
    print(chunk.page_content[:300])
    print("-" * 50)