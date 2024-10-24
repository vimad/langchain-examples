from langchain.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.indexing import index
from langchain_postgres.vectorstores import PGVector

from shared import embedding_model

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
vectorstore = PGVector(embeddings=embedding_model, connection=connection, collection_name='rm-demo')
namespace = f"postgres/langchain"
record_manager = SQLRecordManager(
    namespace, db_url=connection
)
record_manager.create_schema()

doc1 = Document(page_content="kitty", metadata={"source": "kitty.txt"})
doc2 = Document(page_content="doggy", metadata={"source": "doggy.txt"})

def _clear():
    """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")

# ----------NONE------------------------------
print("------clean up mode - None----------")
_clear()
result = index(
    [doc1, doc1, doc1, doc1, doc1],
    record_manager,
    vectorstore,
    cleanup=None,
    source_id_key="source",
)
print(result)

result2 = index([doc1, doc2], record_manager, vectorstore, cleanup=None, source_id_key="source")
print(result2)

# ----------INCREMENTAL---------------
print("------clean up mode - incremental----------")
_clear()
result3 = index(
    [doc1, doc2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
result4 = index(
    [doc1, doc2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
changed_doc_2 = Document(page_content="puppy", metadata={"source": "doggy.txt"})
result5 = index(
    [changed_doc_2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print(result3)
print(result4)
print(result5)