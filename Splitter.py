from langchain_text_splitters import CharacterTextSplitter

text = '''Iran, officially the Islamic Republic of Iran,[d] historically known as Persia, is a country in West Asia. It borders Iraq to the west, Turkey, Azerbaijan, and Armenia to the northwest, the Caspian Sea to the north, Turkmenistan to the northeast, Afghanistan to the east, Pakistan to the southeast, and the Gulf of Oman and the Persian Gulf to the south. With a population of over 92 million, Iran ranks 17th globally in both geographic size and population. It is divided into five regions with 31 provinces. Tehran is the nation's capital and largest city and serves as its primary economic centre.'''
# Define a text splitter that splits on the '.' character
text_splitter = CharacterTextSplitter(
    separator='.',
    chunk_size=75,
    chunk_overlap=10
)

# Split the text using text_splitter
chunks = text_splitter.split_text(text)
print(chunks)
print([len(chunk) for chunk in chunks])