from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient


tavily=TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over a internet
    Args:
        query: The query to search for.
    Returns:
        The search results for the query.
    """
    
    print(f"Searching for {query}...")
    response = tavily.search(query, max_results=2)
    results = response.get("results", [])
    return "\n\n".join(
        f"Title: {result.get('title', '')}\n"
        f"URL: {result.get('url', '')}\n"
        f"Summary: {result.get('content', '')[:600]}"
        for result in results
    )

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

tools = [search]
agent = create_agent(llm, tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 2 job postings for Palantir foundry developer in Bangalore area on linkedin")]})
    print(result)
    
if __name__ == "__main__":
    main()