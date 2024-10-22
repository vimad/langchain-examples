from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from shared import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector

raw_documents = TextLoader('./test.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector.from_documents(documents, embedding_model, connection=connection, collection_name='demo1')
db.add_documents([
  Document(
      page_content="there are cats in the pond",
      metadata={"location": "pond", "topic": "animals"},
  ),
  Document(
      page_content="ducks are also found in the pond",
      metadata={"location": "pond", "topic": "animals"},
  ),
], ids=[1, 2])

result = db.similarity_search("pond", k=4)
# db.delete_collection()
print(result)