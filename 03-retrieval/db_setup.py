from langchain_postgres import PGVector
from shared import embedding_model

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector(embeddings=embedding_model, connection=connection, collection_name='rt-demo')
retriever = db.as_retriever()