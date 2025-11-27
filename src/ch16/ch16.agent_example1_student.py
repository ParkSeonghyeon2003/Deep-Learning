from langchain_tavily import TavilySearch
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import os

# Tavily API 키 설정 (환경 변수 사용을 권장)
os.environ["TAVILY_API_KEY"] = "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" 

# Tavily 검색 리트리버 테스트  ---
retriever = TavilySearchAPIRetriever(k=3)
result = retriever.invoke("2024년 한국 뮤지컬 시카고의 주연 배우들은 누구인가요?")
print(result)

# LLM 모델 설정
model = ChatOllama(model="PetrosStav/gemma3-tools:4b")

# 사용할 도구 설정
search = TavilySearch(max_results=2)
tools = [search]

# 에이전트 생성
agent = create_agent(
    model=model, 
    tools=tools, 
    system_prompt="You are a helpful assistant"
)

# 에이전트 테스트 ---
question = "조지 워싱턴과 루즈벨트 대통령의 어린시절을 비교해서 한국어로 설명해줘."

# 일괄 출력 (시간 오래걸림) ----
result = agent.invoke({
    "messages": [
        {"role":"user", "content":question}
    ]
})

print(result)
print(result["messages"][1])

# 스트리밍 출력 ---
for token, metadata in agent.stream(  
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="messages",
):
    print(token.content_blocks[0]['text'],  end="", flush=True)
