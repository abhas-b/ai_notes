from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
# The following are used for generating structured output

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")
react_prompt = hub.pull("hwchase17/react")
structured_llm = llm.with_structured_output(AgentResponse)

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
).partial(format_instructions="")


agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instructions,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
chain = agent_executor | extract_output | structured_llm


def main():
    print("Hello from 3-react-search-agent! - All imports done")
    query = """search for 3 job postings for an ai engineer using langchain in bangalore or bengaluru on linkedin and list their details."""
    result = chain.invoke(input={"input": query})
    print(result)


if __name__ == "__main__":
    main()
