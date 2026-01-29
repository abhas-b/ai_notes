from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field
import os

load_dotenv(dotenv_path="../.env")

class Source(BaseModel):
    """Schema for the source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response"""

    answer:str = Field(description="The agent answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources to generate the answer.")



llm = ChatOpenAI(temperature=0, model='gpt-4o-mini')
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from 2-basic-search-agent!")
    query = """search for 3 job postings for an ai engineer using langchain in bangalore or bengaluru and list their details."""
    result = agent.invoke({"messages":[HumanMessage(content=query)]})
    print(result)


if __name__ == "__main__":
    main()
