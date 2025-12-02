
# stdio 연결을 위한 서버 매개변수 생성
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import asyncio

# LLM 모델 설정
model = ChatOllama(model="PetrosStav/gemma3-tools:4b")

# 서버 매개변수 구성
server_params = StdioServerParameters(
    command="python",
    # 서버 파일 경로 지정
    args=["D:/OneDrive - 단국대학교/2025강의/딥러닝/source_code/chap17_mcp_server.py"],
)

async def run_agent(query):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 연결 초기화
            await session.initialize()
            
            # LangChain 형식으로 MCP 도구 로드
            tools = await load_mcp_tools(session)
            
            # 에이전트를 생성하고 실행
            agent = create_agent(
                model=model, 
                tools=tools, 
                system_prompt="You are a helpful assistant"
            )
            agent_response = await agent.ainvoke({"messages": query})
            return agent_response

# 비동기 함수 실행
if __name__ == "__main__":
    query = "what's (3 + 5) x 12?"
    result = asyncio.run(run_agent(query))
    print(result)