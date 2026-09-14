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

# def ask_query():

#     chat_history='' 
#     while True:
#         query = input("Enter your query: ")
#         print(type(query))
#         if query.lower()=='exit':
#             break
#         if not chat_history:
#             print("2nd block")
#             query = query
#         else:
#             print("3rd block")
#             new_content = query+chat_history
#             query_formater = [
#                         SystemMessage(
#                             content='You are a query maker who takes current query and chat history and convert the query based on the chat history. put emphaisis on current query only. example it chat history contains info like quey was when ws google founded, then asnwer ''it was founded in 1998'' and another question ''who are its founder' 'and answer' 'sergery brin and laary page founded company in 1998' 'for hir question if i ask what its revenue then new query is whats google revuee not who is google founder,its reveune when was founded. put emphaisi on current question only'
#                         ),
#                         HumanMessage(
#                             content=new_content
#                         )
#                     ]

#             query=llm.invoke(query_formater).content

#             print("Formated query..")
#             print(query)

#         chat_history+=f'query: {query}'
#         print(f"\nOriginal Query: {query}")

#         docs = retriever.invoke(query)

#         context=""

#         for i, doc in enumerate(docs, start=1):

#             source = doc.metadata.get("source")
#             content = doc.page_content

#             context += f"""
#         Source: {source}
#         Content: {content}

#         """

#         prompt = f"""
#         Answer the question using ONLY the information provided
#         in the context.Also provide from which source you took the
#         content from.Do not invent Sources.

#         Context:
#         {context}

#         Question:
#         {query}

#         If the answer is not present in the context, say:

#         "I don't know based on the provided information."
#         """

#         response = llm.invoke(prompt)

#         print("\n================ FINAL ANSWER ================")

#         print(response.content)
#         chat_history+=f'response: {response}'


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
Answer the question using ONLY the information provided
in the context.

Also provide the source from which the information was taken.
Do not invent sources.

Context:
{context}

Question:
{formatted_query}

If the answer is not present in the context, say:

"I don't know based on the provided information."
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

