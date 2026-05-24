from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import os

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as exc:
    HuggingFaceEmbeddings = None
    HUGGINGFACE_IMPORT_ERROR = exc


load_dotenv()


def load_document(docs_path="docs"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exits , please create one and add your files.")
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(
            f"The directory {docs_path} does not contain any text file , please add some text file.")
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f"  Source:{doc.metadata['source']}")
        print(f"  Content Length:{len(doc.page_content)} characters")
        print(f"  Content Preview:{doc.page_content[:100]}... ")
        print(f"  Metadata: {doc.metadata}")
    return documents


def split_document(documents, chunk_size=800, chunk_overlap=0):
    print("splitting document into chunks")
    text_splitter = CharacterTextSplitter(
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size
    )
    chunks = text_splitter.split_documents(documents)
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n chunk{i+1}")
            print(f"  Source:{chunk.metadata['source']}")
            print(f"  content_lenght: {len(chunk.page_content)} characters")
            print(f"  Content :{chunk.page_content}")
            print("-"*50)
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("creating embeding and storing them in vector database")
    if HuggingFaceEmbeddings is None:
        raise ImportError(
            "langchain_huggingface is not installed. Install it with: pip install langchain-huggingface sentence-transformers"
        ) from HUGGINGFACE_IMPORT_ERROR

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("--creating vector store---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("-------Finished creating the vector store ------")
    print(
        f" vector store created succesfully and stored in {persist_directory}")
    return vectorstore


def main():
    print("main function ")
    documents = load_document("docs")
    chunks = split_document(documents)
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()
