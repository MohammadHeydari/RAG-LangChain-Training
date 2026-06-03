from langchain_community.document_loaders import UnstructuredHTMLLoader

# Create a document loader for unstructured HTML
loader = UnstructuredHTMLLoader("data/dc.html")

# Load the document
data = loader.load()

# Print the first document's content
print(data[0].page_content)

# Print the first document's metadata
print(data[0].metadata)