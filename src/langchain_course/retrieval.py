
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

if __name__ == '__main__':

    print("Connecting to Pinecone...")

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(os.environ.get("INDEX_NAME"))

    print("Initializing Groq...")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    query = input("\nEnter your question: ")

    print("\nSearching for relevant documents...")

    results = index.documents.search(
        namespace="default",
        top_k=3,
        score_by=[
            {
                "type": "embed",
                "field": "chunk_text",
                "query": query
            }
        ],
        include_fields=["chunk_text"]
    )

    context = "\n\n".join(
        map(lambda match: match.chunk_text, results.matches)
    )

    if not context:
        print("No relevant documents found.")
    else:

        prompt = PromptTemplate.from_template(
            """
            Answer the question using only the following context.
            If the answer is not available in the context,
            say you don't know.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        rag_chain = prompt | llm

        print("\nGenerating answer...")

        response = rag_chain.invoke({
            "context": context,
            "question": query
        })

        print("\n" + "=" * 50)
        print("QUESTION:")
        print(query)

        print("\nANSWER:")
        print(response.content)
        print("=" * 50)
