from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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

    chat_history = []

    while True:

        original_query = input("Enter your query: ")

        if original_query.lower() == 'exit':
            break

        # ---------------- QUERY REWRITING ----------------

        if not chat_history:

            formatted_query = original_query

        else:

            query_formater = [

                SystemMessage(
                    content="""
You are a query rewriter.

Your job is to convert the CURRENT user query
into a standalone query using the conversation history.

Focus mainly on the CURRENT query.

Use previous conversation only to resolve references
like "it", "they", "this", "that", "how many", etc.

Return ONLY the rewritten query.
Do not write "New query:".
Do not explain anything.
"""
                ),

                *chat_history,

                HumanMessage(
                    content=original_query
                )
            ]

            formatted_query = llm.invoke(query_formater).content

        print("\nFormatted Query:")
        print(formatted_query)

        # ---------------- RETRIEVAL ----------------

        docs = retriever.invoke(formatted_query)

        context = ""

        for i, doc in enumerate(docs, start=1):

            print("*"*60)

            print(f'Document {i}')



            source = doc.metadata.get("source")

            print(source)
            content = doc.page_content
            print(content)

            print("*"*60)

            context += f"""
Source: {source}
Content: {content}

"""

        # ---------------- ANSWER GENERATION ----------------

        prompt = f"""

Question:

{formatted_query}

Answer the question using ONLY the information provided in the context.

Rules:
1. If the context contains information that directly or clearly supports the answer, answer the question using that information.
2. Do not require the context to use the exact same wording as the question. Treat synonyms and closely related terms as equivalent when the meaning is clear.
3. If the question asks for a specific detail that is not stated or cannot be reasonably determined from the context, say:
"I don't know based on the provided information."
4. Do not invent, assume, or add information that is not supported by the context.
5. Prefer a concise answer and mention the relevant source when possible.

Context:
{context}
"""

        response = llm.invoke(prompt)

        print("\n================ FINAL ANSWER ================")
        print(response.content)

        # ---------------- UPDATE CHAT HISTORY ----------------

        chat_history.append(
            HumanMessage(content=original_query)
        )

        chat_history.append(
            AIMessage(content=response.content)
        )
ask_query()

