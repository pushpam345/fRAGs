from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

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

final_combined_input = f'''based on following document , please answer this question:{query} 

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}


please provide a clear , helpful answer using only the information in these documents , if you can't find the answer in documents, say " I dont have enough information to answer the question based on the provided document "


'''
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)
messages = [
    SystemMessage(content="you are a helpful assistant ."),
    HumanMessage(content=final_combined_input)
]
result = model.invoke(messages)

print("----generated response---")
print("Content only : ")
print(result.content)
