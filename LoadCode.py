from langchain_community.document_loaders import PythonLoader

# Create a document loader for rag.py and load it
loader = PythonLoader('data/LCEL.py')

python_data = loader.load()
print(python_data[0])