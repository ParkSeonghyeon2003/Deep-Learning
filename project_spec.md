# 프로젝트 명세서: Local Insight Agent (Korean Tech Reporter)

## 1. 프로젝트 개요

- **목표:** 사용자가 한국어로 입력한 주제에 대해 Tavily API로 심층 검색을 수행하고, 로컬 LLM(Ollama)이 정보를 분석하여 구조화된 한국어 보고서를 작성하는 Agent 앱 개발.
- **핵심 가치:** 1. 외부 클라우드 LLM이 아닌 로컬 GPU 활용 (Privacy & Cost). 2. 단순 검색 결과 나열이 아닌 '종합 요약 및 인사이트' 제공.
- **마감 기한:** 8일 (프로토타입 완성 필수).

## 2. 개발 환경 및 스펙

- **OS:** Windows 11 Desktop
- **Language:** Python 3.10+
- **Dependency Manager:** Poetry (필수)
- **Hardware Constraint:** NVIDIA RTX 2070 (8GB VRAM) -> 양자화된 로컬 모델 사용 필수.
- **LLM Server:** Ollama (Model: `llama3.1` 8B)
- **Search Tool:** Tavily API (Agent 전용 검색 도구)

## 3. 파일 구조 및 역할 (File Structure)

- `src/main.py`: Streamlit 기반의 웹 UI. 사용자 입력 수신 및 결과 표시.
- `src/llm.py`: `langchain_ollama`를 사용하여 로컬 Llama 3.1 모델을 초기화. (Temperature=0 설정)
- `src/agent.py`: `TavilySearchResults` 도구를 사용하여 검색하고, LLM과 결합하여 답변을 생성하는 LangGraph 혹은 Chain 로직.
- `src/utils.py`: 필요한 경우 문자열 파싱이나 파일 저장 기능 담당.
- `.env`: `TAVILY_API_KEY` 관리.

## 4. 기능 요구 사항 (Functional Requirements)

1. **한국어 최적화:** 모든 프롬프트(System Prompt)는 에이전트가 "한국어"로 답변하도록 강력하게 지시해야 함.
2. **검색 및 통합:** - 사용자가 하나의 주제(예: "트랜스포머 모델의 발전사")를 입력.
   - Tavily로 최소 3개 이상의 문서를 검색.
   - 내용을 종합하여 서론-본론-결론이 있는 마크다운 리포트로 출력.
3. **UI/UX:**
   - 진행 상황을 알 수 있는 로딩 인디케이터(`st.spinner`, `st.status`) 필수.
   - 결과물 하단에 '참고 문헌(Reference URLs)' 리스트 표시.
4. **에러 처리:**
   - Ollama 서버가 꺼져있거나, 검색 결과가 없을 때 적절한 에러 메시지 출력.

## 5. 라이브러리 스택

- `streamlit`
- `langchain`, `langchain-community`, `langchain-ollama`
- `langchain-tavily`
- `python-dotenv`
