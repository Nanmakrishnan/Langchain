import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = TextLoader("mediumblog1.txt", encoding="utf-8")

if __name__ == '__main__':
    print("Ingesting....")

    document = loader.load()

    print("Splitting....")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    texts = text_splitter.split_documents(document)

    print(f"Number of text chunks: {len(texts)}")

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(os.environ.get("INDEX_NAME"))

    records = [
        {
            "_id": str(i),
            "chunk_text": text.page_content
        }
        for i, text in enumerate(texts)
    ]

    print("Uploading to Pinecone...")

    index.documents.upsert(
        namespace="default",
        documents=records
    )

    print("Ingestion complete!")