from langchain_community.document_loaders import PythonLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


loader = PythonLoader("LCEL.py")
python_data = loader.load()

print(f"Loaded {len(python_data)} document(s) from LCEL.py")


python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=100
)


chunks = python_splitter.split_documents(python_data)

print(f"Number of chunks: {len(chunks)}")


for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n")