from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings_model = HuggingFaceEmbeddings(
               model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME='Nimbus Retail Q&A'
db = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings_model
    )

print("Connected to existing vector database")

retriever = db.as_retriever(
    search_kwargs={
        "k": 3
    }
)

def ask_query():

    query = input("Enter your query: ")

    print(f"\nOriginal Query: {query}")

    docs = retriever.invoke(query)

    context=""

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get("source")
        content = doc.page_content

        context += f"""
    Source: {source}
    Content: {content}

    """

    prompt = f"""
    Answer the question using ONLY the information provided
    in the context.Also provide from which source you took the
    content from.Do not invent Sources.

    Context:
    {context}

    Question:
    {query}

    If the answer is not present in the context, say:

    "I don't know based on the provided information."
    """

    response = llm.invoke(prompt)

    print("\n================ FINAL ANSWER ================")

    print(response.content)

ask_query()

