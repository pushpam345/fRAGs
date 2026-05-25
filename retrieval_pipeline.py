from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

persistant_directory = "db/chroma_db"
db = Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
query = "what was nvidia's first graphic accelerator called ?"

retriever = db.as_retriever(
    # search_type="similarty_score_threshhold",
    search_kwargs={
        "k": 5,
        # "score_threshold": 0.3
    }
)
relevant_docs = retriever.invoke(query)
for i, doc in enumerate(relevant_docs):
    print(f" Document{i+1}:\n{doc.page_content}\n")
