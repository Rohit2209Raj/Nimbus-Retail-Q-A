import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType
)
load_dotenv()

def load_documents():
    loader = DirectoryLoader(
        path='docs',
        glob='*.md',
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    print(f'Loaded {len(documents)} Documents from docs')

    return documents


def make_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap = 50
    )

    chunks = text_splitter.split_documents(documents)

    print(f'Created {len(chunks)} Chunks from docs')



    return chunks

def make_database(chunks):

    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    qdrant_client=QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    print(f'Connected toQdrant Cloud!')

    COLLECTION_NAME = 'Nimbus Retail Q&A'
    EMBEDDING_SIZE = 384

    embedding_model=HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    print(f'Embedding model loaded')


    if qdrant_client.collection_exists(COLLECTION_NAME):
        print(f'Deleting existing collection:{COLLECTION_NAME}')
        qdrant_client.delete_collection(COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )

    print(f"Created collection: {COLLECTION_NAME}")

    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.source",
        field_schema=PayloadSchemaType.KEYWORD
    )

    print("Created payload index: metadata.source")

    documents = [
    Document(
        page_content=item.page_content,
        metadata={
            "source": os.path.basename(item.metadata["source"]),
            "chunk_id": f"chunk_{i}"
        }
    )
    for i, item in enumerate(chunks)
]

    print("Converted Chunks into LangChain Documents")

    db=QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )

    db.add_documents(documents)

    print(f"Stored {len(documents)} documents in Qdrant")



document = load_documents()

chunks = make_chunks(document)

make_database(chunks)

