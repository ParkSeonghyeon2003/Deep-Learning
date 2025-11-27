from langchain_tavily import TavilySearch
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from datetime import datetime
from langchain.tools import tool
import os

# Tavily API 키 설정 (환경 변수 사용을 권장)
os.environ["TAVILY_API_KEY"] = "your key" 

# Tavily 검색 리트리버 테스트  ---
retriever = TavilySearchAPIRetriever(k=3)
result = retriever.invoke("2024년 한국 뮤지컬 시카고의 주연 배우들은 누구인가요?")
print(result)

# LLM 모델 설정
model = ChatOllama(model="PetrosStav/gemma3-tools:4b")

# 사용할 도구 설정
search = TavilySearch(max_results=2)

@tool
def get_time_tool(input_str: str) -> str:
    """
    현재 시간을 반환합니다. 입력은 사용되지 않습니다.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


tools = [search, get_time_tool]

# 에이전트 생성
agent = create_agent(
    model=model, 
    tools=tools, 
    system_prompt="You are a helpful assistant"
)

# 스트리밍 출력 ---
def get_answer(question):
    print("Answer :")
    for token, metadata in agent.stream(  
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages",
    ):
        if len(token.content_blocks) == 0 :
            continue
        if len(token.content_blocks[0]) != 2 :
            continue

        if len(token.content_blocks[0]['text']) < 20 :
            print(token.content_blocks[0]['text'],  end="", flush=True)
    print("\n")

# 에이전트 테스트 ---
question_1 = "내일 용인 날씨를 알려주시오"
question_2 = "지금 몇시야?"

get_answer(question_1)
get_answer(question_2)

