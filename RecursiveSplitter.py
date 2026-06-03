from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader('data/Designing_Data_Intensive_Applications_The_Big_Ideas_Behind_Reliable.pdf')

document = loader.load()

# Define a text splitter that splits recursively through the character list
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n', '.', ' ', ''],
    chunk_size=75,
    chunk_overlap=10
)

# Split the document using text_splitter
chunks = text_splitter.split_documents(document)
print(chunks)
print([len(chunk.page_content) for chunk in chunks])