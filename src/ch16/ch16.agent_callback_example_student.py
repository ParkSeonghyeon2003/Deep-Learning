# Import relevant functionality
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import os

os.environ["TAVILY_API_KEY"] = "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"

# 사용할 도구 설정
search = TavilySearch(max_results=2)
tools = [search]

# Create the agent
model = ChatOllama(model="PetrosStav/gemma3-tools:4b")
tools = [search]

# 에이전트 생성
agent = create_agent(
    model=model, 
    tools=tools, 
    system_prompt="You are a helpful assistant"
)

# 처리 단계별 모니터링
question = "대한민국의 AI 투자관련 뉴스를 검색해 주세요."

for chunk in agent.stream(  
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="updates"
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")
 