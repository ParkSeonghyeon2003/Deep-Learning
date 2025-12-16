# Tech Trend Researcher

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/ParkSeonghyeon2003/Tech_Trend_Researcher)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-1.1.1+-orange.svg)](https://langchain.com)

로컬 LLM(Ollama)과 Tavily API를 활용한 AI 기술 리포트 자동 생성 시스템

## 🎯 프로젝트 개요

사용자가 입력한 주제에 대해 Tavily API로 심층 검색을 수행하고, 로컬에서 실행되는 Llama 3.1 모델이 정보를 분석하여 구조화된 한국어 보고서를 자동으로 작성합니다.

### 핵심 가치

- **프라이버시**: 외부 클라우드 LLM이 아닌 로컬 GPU 활용
- **비용 절감**: API 호출 비용 최소화
- **인사이트 제공**: 단순 검색 결과 나열이 아닌 종합 분석 및 인사이트 제공

## 🛠️ 기술 스택

- **언어**: Python 3.10+
- **의존성 관리**: pip 또는 Poetry(선택)
- **웹 프레임워크**: Streamlit
- **LLM**: Ollama (Llama 3.1 8B)
- **검색 API**: Tavily
- **AI 프레임워크**: LangChain (최신 버전)
  - `langchain (>=1.1.2,<2.0.0)`
  - `langchain-community (>=0.4.1,<0.5.0)`
  - `langchain-ollama (>=1.0.0,<2.0.0)`
  - `langchain-tavily (>=0.2.13,<0.3.0)`
  - LCEL (LangChain Expression Language) 파이프라이닝 사용 (`prompt | llm | parser`)

## 📋 요구사항

### 하드웨어

- NVIDIA GPU (최소 8GB VRAM 권장)
- 예: RTX 2070 이상

### 소프트웨어

- Python 3.10 이상
- Poetry (선택)
- Ollama

## 🚀 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/ParkSeonghyeon2003/Deep-Learning.git
```

### 2. 의존성 설치

아래 두 가지 방법 중 하나를 선택하세요.

방법 A) pip (권장)

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows (Git Bash)
pip install -e .
```

방법 B) Poetry (선택)

현재 `pyproject.toml`은 PEP 621 형식을 사용합니다. Poetry를 사용할 경우 최신 버전이 필요할 수 있습니다. 문제가 발생하면 pip 방식을 권장합니다.

```bash
poetry install
```

### 3. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (https://ollama.ai)
# Windows: 웹사이트에서 다운로드

# 모델 다운로드
ollama pull llama3.1

# Ollama 서버 실행
ollama serve
```

### 4. 환경 변수 설정

`.env` 파일에 Tavily API Key를 설정하세요:

```env
TAVILY_API_KEY=your_tavily_api_key_here
```

Tavily API Key는 [https://tavily.com](https://tavily.com)에서 발급받을 수 있습니다.

### 5. 애플리케이션 실행

pip 방식

```bash
streamlit run src/main.py
```

Poetry 방식

```bash
poetry run streamlit run src/main.py
```

브라우저에서 자동으로 열립니다 (보통 `http://localhost:8501`)

## 📁 프로젝트 구조

```
Tech_Trend_Researcher/
├── src/
│   ├── main.py            # Streamlit UI (진입점)
│   ├── agent.py           # 검색 및 리포트 생성 로직
│   ├── llm.py             # Ollama LLM 초기화
│   ├── utils.py           # 유틸리티 함수 (저장/검증 포함)
│   └── logging_utils.py   # 터미널 예쁜 로그 유틸리티
├── .env                   # API Key 설정 (선택)
├── pyproject.toml         # 프로젝트 메타데이터 및 의존성
└── README.md              # 프로젝트 문서
```

## 💡 사용 방법

1. **주제 입력**: 리서치할 기술 주제나 트렌드를 입력합니다

   - 예: "트랜스포머 모델의 발전사"
   - 예: "RAG 기술 동향"

2. **보고서 생성**: "보고서 생성" 버튼을 클릭합니다

3. **결과 확인**: AI가 검색하고 분석한 구조화된 리포트를 확인합니다

   - 서론, 본론, 결론 형식
   - 참고 문헌 자동 포함

4. **저장 (선택사항)**: 생성된 리포트를 마크다운 파일로 저장할 수 있습니다

## 🔧 트러블슈팅

### Ollama 연결 오류

```bash
# Ollama 서버 실행 확인
ollama serve

# 모델 다운로드 확인
ollama list

# 모델 재다운로드
ollama pull llama3.1
```

### API Key 오류

- Tavily API Key가 올바른지 확인
- [https://tavily.com](https://tavily.com)에서 새 키 발급

### 검색 결과 없음

- 인터넷 연결 상태 확인
- 다른 키워드나 주제로 재시도

## 📝 주요 기능

### ✨ 한국어 최적화

- 모든 프롬프트에 한국어 답변 강제 지시
- 한국어로 자연스러운 리포트 생성

### 🔍 심층 검색

- Tavily API의 advanced 검색 모드 사용
- 최대 3개의 관련 문서 검색

### 📊 구조화된 리포트

- 서론-본론-결론 형식
- 마크다운 형식으로 가독성 높은 출력
- 참고 문헌 자동 포함

### 💾 리포트 저장

- 생성된 리포트를 Markdown 파일로 저장

### 🧾 예쁜 터미널 로그 (개발자용)

- 단계별 진행 상황을 아이콘/컬러로 출력 (`logging_utils.py`)
- 환경 변수로 제어 가능
  - `PRETTY_LOG=1|0` (기본 1)
  - `LOG_LEVEL=DEBUG|INFO|WARN|ERROR` (기본 INFO)

### 🎨 사용자 친화적 UI

- 실시간 진행 상황 표시
- 직관적인 인터페이스
- 상세한 에러 메시지 및 해결 방법 안내

## 🔒 프라이버시 & 보안

- LLM 추론은 로컬 Ollama에서 수행
- 외부 클라우드 LLM으로 데이터 전송 없음
- Tavily API에는 검색 쿼리와 필수 메타데이터만 전송

## 🧪 빠른 확인(로컬 실행 예)

```bash
# 가상환경 생성 및 활성화 (Windows Git Bash)
python -m venv .venv
source .venv/Scripts/activate

# 의존성 설치
pip install -e .

# 환경 변수 설정 (.env 또는 UI에서 입력 가능)
echo "TAVILY_API_KEY=your_tavily_api_key_here" > .env

# 앱 실행
streamlit run src/main.py
```

## 📄 라이선스

MIT License

## 🙏 감사의 글

- [Ollama](https://ollama.ai) - 로컬 LLM 실행 환경
- [Tavily](https://tavily.com) - AI 에이전트 전용 검색 API
- [LangChain](https://langchain.com) - AI 애플리케이션 프레임워크
- [Streamlit](https://streamlit.io) - 웹 UI 프레임워크


