from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")
# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)

#  create_documents, which accepts a list of strings, rather than the list of documents we had before
python_docs = python_splitter.create_documents([PYTHON_CODE])
print(python_docs)