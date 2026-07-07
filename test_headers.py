import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

@tool
def dummy_tool(query: str) -> str:
    """A dummy search tool."""
    return "42"

def run_test(with_headers):
    headers = {
        "HTTP-Referer": "https://github.com/diegosouzapw/blog-agent",
        "X-Title": "AI Blog Generator Agent",
    } if with_headers else None
    
    print(f"Testing with_headers={with_headers}...")
    llm = ChatOpenAI(
        model="openrouter/openai/gpt-4o-mini",
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0,
        default_headers=headers
    )
    agent = create_react_agent(
        model=llm,
        tools=[dummy_tool],
        prompt="Answer the user using dummy_tool.",
    )
    try:
        res = agent.invoke({"messages": [{"role": "user", "content": "What is the answer?"}]})
        print(f"Success! Response: {res['messages'][-1].content}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    run_test(with_headers=True)
    run_test(with_headers=False)
