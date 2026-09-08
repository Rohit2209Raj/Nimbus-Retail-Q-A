import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from dotenv import load_dotenv

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

document = load_documents()

chunks = make_chunks(document)

