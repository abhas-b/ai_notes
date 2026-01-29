from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


tools = [TavilySearch()]
llm = ChatOpenAI(model='gpt-4o-mini')
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm,tools=tools,prompt=react_prompt,)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

def main():
    print("Hello from 3-react-search-agent! - All imports done")
    query = """search for 3 job postings for an ai engineer using langchain in bangalore or bengaluru on linkedin and list their details."""
    result = chain.invoke(
        input={"input":query}
    )
    print(result)


if __name__ == "__main__":
    main()
