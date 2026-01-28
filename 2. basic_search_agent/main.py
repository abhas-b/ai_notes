from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
# from langchain_tavily import TavilySearch
import os

load_dotenv(dotenv_path="../.env")

tavily_key = os.environ['TAVILY_API_KEY']
tavily = TavilyClient(api_key=tavily_key)

#custom impementation. Can easily be replaced with TavilySearch from Tavily team itself.
@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet to respond to a user query seeking information.

    Args:
        query: the query to search for
    Returns:
        the search result
    """

    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(temperature=0, model='gpt-4o-mini')
tools = [search]
# tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from 2-basic-search-agent!")
    query = """search for 3 job postings for an ai engineer using langchain in bangalore or bengaluru and list their details."""
    result = agent.invoke({"messages":[HumanMessage(content=query)]})
    print(result)


if __name__ == "__main__":
    main()
