# Import library
from langchain_community.document_loaders import PyPDFLoader

# Create a document loader for rag_paper.pdf
loader = PyPDFLoader('data/Designing_Data_Intensive_Applications_The_Big_Ideas_Behind_Reliable.pdf')

# Load the document
data = loader.load()
print(data[0])