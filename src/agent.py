"""
Agent 모듈: Tavily 검색과 LLM을 결합한 리포트 생성 에이전트
"""
import os
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm import get_llm
from utils import format_search_results, extract_urls
from logging_utils import section, step, info, success, search as log_search, llm as log_llm

# .env 파일에서 환경 변수 로드
load_dotenv()


def get_search_tool() -> TavilySearch:
    """
    Tavily 검색 도구를 초기화하여 반환합니다.
    
    Returns:
        TavilySearch: 초기화된 Tavily 검색 도구
    """
    # API Key 확인
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "your_tavily_api_key_here":
        raise ValueError(
            "Tavily API Key가 설정되지 않았습니다. "
            ".env 파일에 TAVILY_API_KEY를 설정하거나 UI에서 입력하세요."
        )
    
    return TavilySearch(
        max_results=3,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True
    )


def create_report_prompt() -> ChatPromptTemplate:
    """
    리포트 생성을 위한 프롬프트 템플릿을 생성합니다.
    
    Returns:
        ChatPromptTemplate: 프롬프트 템플릿
    """
    return ChatPromptTemplate.from_messages([
        ("system", "당신은 IT 전문 기술 블로거입니다. 항상 한국어로 명확하고 논리적으로 답변하세요."),
        ("user", """다음 주제에 대한 검색 결과를 바탕으로 구조화된 한국어 리포트를 작성하세요.

주제: {topic}

검색 결과:
{search_results}

다음 형식으로 리포트를 작성하세요:

# {topic}

## 서론
(주제에 대한 간단한 소개)

## 본론
(검색 결과를 종합하여 핵심 내용을 정리. 여러 섹션으로 나누어 작성)

## 결론
(핵심 요점 정리 및 인사이트)

---

마크다운 형식으로 작성하고, 전문적이고 읽기 쉽게 구성하세요.""")
    ])


def generate_report(topic: str) -> Dict[str, Any]:
    """
    주제에 대한 리서치를 수행하고 리포트를 생성합니다.
    
    Args:
        topic: 리서치 주제
        
    Returns:
        dict: {
            "report": 생성된 리포트 본문,
            "sources": 참고한 URL 리스트,
            "raw_results": 원본 검색 결과 (선택사항)
        }
        
    Raises:
        ValueError: API Key가 유효하지 않을 때
        ConnectionError: Ollama 서버에 연결할 수 없을 때
        Exception: 기타 예상치 못한 오류
    """
    try:
        section("리포트 생성 시작", icon="rocket")
        info("입력 주제", kv={"topic": topic[:40] + ("..." if len(topic) > 40 else "")})
        # 검색 도구 초기화
        try:
            search_tool = get_search_tool()
            success("검색 도구 준비 완료", kv={"provider": "Tavily"})
        except ValueError as e:
            raise ValueError(f"[검색 도구 초기화 실패] {str(e)}")
        except Exception as e:
            raise Exception(f"[검색 도구 초기화 중 예상치 못한 오류] {type(e).__name__}: {str(e)}")
        
        # 검색 수행
        try:
            log_search("검색 수행", kv={"query_len": str(len(topic))})
            search_response = search_tool.invoke(topic)
        except Exception as e:
            raise Exception(
                f"[Tavily 검색 실패] 주제: '{topic}'\n"
                f"오류 타입: {type(e).__name__}\n"
                f"오류 내용: {str(e)}\n"
                f"API 키가 유효한지, 네트워크 연결이 정상인지 확인하세요."
            )
        
        # 검색 응답 타입 검증
        if not search_response:
            raise ValueError(
                f"[검색 결과 없음] '{topic}'에 대한 검색 결과를 찾을 수 없습니다.\n"
                f"다른 키워드로 시도하거나 더 구체적인 주제를 입력해보세요."
            )
        
        # TavilySearch가 딕셔너리를 반환하는 경우 처리
        if isinstance(search_response, dict):
            # 'results' 키에서 실제 검색 결과 추출
            if 'results' not in search_response:
                raise Exception(
                    f"[검색 결과 형식 오류] Tavily 응답에 'results' 키가 없습니다.\n"
                    f"응답 키 목록: {list(search_response.keys())}\n"
                    f"응답 내용: {str(search_response)[:300]}"
                )
            search_results = search_response['results']
        elif isinstance(search_response, list):
            # 리스트를 직접 반환하는 경우
            search_results = search_response
        else:
            raise Exception(
                f"[검색 결과 형식 오류] Tavily가 예상치 못한 형식으로 결과를 반환했습니다.\n"
                f"반환 타입: {type(search_response).__name__}\n"
                f"반환 내용 (처음 200자): {str(search_response)[:200]}\n"
                f"예상 타입: Dict 또는 List[Dict]"
            )
        
        # 검색 결과가 리스트인지 확인
        if not isinstance(search_results, list):
            raise Exception(
                f"[검색 결과 형식 오류] 추출된 검색 결과가 리스트 형식이 아닙니다.\n"
                f"결과 타입: {type(search_results).__name__}\n"
                f"결과 내용: {str(search_results)[:200]}"
            )
        
        # 검색 결과가 비어있는지 확인
        if len(search_results) == 0:
            raise ValueError(
                f"[검색 결과 없음] '{topic}'에 대한 검색 결과가 비어있습니다.\n"
                f"다른 키워드로 시도하거나 더 구체적인 주제를 입력해보세요."
            )
        success("검색 완료", kv={"results": str(len(search_results))})
        
        # URL 추출
        try:
            sources = extract_urls(search_results)
            info("출처 수집", kv={"urls": str(len(sources))})
        except Exception as e:
            raise Exception(
                f"[URL 추출 실패] 검색 결과에서 URL을 추출하는 중 오류 발생\n"
                f"오류 타입: {type(e).__name__}\n"
                f"오류 내용: {str(e)}\n"
                f"검색 결과 형식: {type(search_results)}\n"
                f"검색 결과 샘플: {str(search_results[0]) if search_results else 'N/A'}"
            )
        
        # 검색 결과 포맷팅
        try:
            step("결과 포맷팅")
            formatted_results = format_search_results(search_results)
        except Exception as e:
            raise Exception(
                f"[검색 결과 포맷팅 실패] 검색 결과를 포맷팅하는 중 오류 발생\n"
                f"오류 타입: {type(e).__name__}\n"
                f"오류 내용: {str(e)}\n"
                f"검색 결과 개수: {len(search_results) if isinstance(search_results, list) else 'N/A'}\n"
                f"검색 결과 타입: {type(search_results)}\n"
                f"첫 번째 결과 타입: {type(search_results[0]) if search_results else 'N/A'}\n"
                f"첫 번째 결과 내용: {str(search_results[0])[:200] if search_results else 'N/A'}"
            )
        
        # LLM 및 프롬프트 초기화
        try:
            log_llm("LLM 준비 중")
            llm = get_llm()
        except ConnectionError as e:
            raise ConnectionError(
                f"[LLM 연결 실패] Ollama 서버에 연결할 수 없습니다.\n"
                f"오류 내용: {str(e)}\n"
                f"해결 방법:\n"
                f"1. Ollama가 실행 중인지 확인 (ollama serve)\n"
                f"2. 포트가 올바른지 확인 (기본: 11434)\n"
                f"3. 방화벽 설정 확인"
            )
        except Exception as e:
            raise Exception(
                f"[LLM 초기화 실패] LLM을 초기화하는 중 오류 발생\n"
                f"오류 타입: {type(e).__name__}\n"
                f"오류 내용: {str(e)}"
            )
        
        try:
            prompt = create_report_prompt()
        except Exception as e:
            raise Exception(
                f"[프롬프트 생성 실패] 프롬프트 템플릿 생성 중 오류 발생\n"
                f"오류 타입: {type(e).__name__}\n"
                f"오류 내용: {str(e)}"
            )
        
        # 체인 구성 및 실행
        try:
            chain = prompt | llm | StrOutputParser()
            step("LLM 체인 실행")
            report = chain.invoke({
                "topic": topic,
                "search_results": formatted_results
            })
        except Exception as e:
            raise Exception(
                f"[리포트 생성 실패] LLM 체인 실행 중 오류 발생\n"
                f"주제: '{topic}'\n"
                f"오류 타입: {type(e).__name__}\n"
                f"오류 내용: {str(e)}\n"
                f"검색 결과 길이: {len(formatted_results)} 문자\n"
                f"참고: LLM 모델이 설치되어 있는지 확인하세요 (ollama list)"
            )
        
        success("리포트 생성 완료")
        return {
            "report": report,
            "sources": sources
        }
        
    except (ValueError, ConnectionError) as e:
        # 이미 상세한 메시지가 포함된 예외는 그대로 전달
        raise
    except Exception as e:
        # 예상치 못한 최상위 오류
        raise Exception(
            f"[예상치 못한 오류] 리포트 생성 중 알 수 없는 오류가 발생했습니다.\n"
            f"오류 타입: {type(e).__name__}\n"
            f"오류 내용: {str(e)}\n"
            f"주제: '{topic}'\n"
            f"디버깅을 위해 전체 스택 트레이스를 확인하세요."
        )
