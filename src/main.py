"""
Streamlit 기반 웹 UI: AI Tech Report Agent
"""
import streamlit as st
import os
from dotenv import load_dotenv

from agent import generate_report
from utils import validate_api_key

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="AI Tech Report Agent",
    page_icon="📊",
    layout="wide"
)

# 타이틀
st.title("📊 AI Tech Report Agent")
st.markdown("**로컬 LLM과 Tavily 검색을 활용한 기술 리포트 자동 생성 시스템**")
st.divider()

# 세션 상태 초기화
if "report_data" not in st.session_state:
    st.session_state["report_data"] = None  # {report:str, topic:str, sources:list[str]}

# 사이드바: API Key 관리
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 환경 변수에서 API Key 확인
    env_api_key = os.getenv("TAVILY_API_KEY")
    
    if env_api_key and validate_api_key(env_api_key):
        st.success("✅ Tavily API Key 로드됨 (.env)")
        api_key = env_api_key
    else:
        st.warning("⚠️ .env 파일에 API Key가 없습니다.")
        api_key = st.text_input(
            "Tavily API Key를 입력하세요:",
            type="password",
            help="https://tavily.com 에서 API Key를 발급받으세요."
        )
        
        if api_key and validate_api_key(api_key):
            os.environ["TAVILY_API_KEY"] = api_key
            st.success("✅ API Key 설정 완료")
        elif api_key:
            st.error("❌ 유효하지 않은 API Key 형식입니다.")
    
    st.divider()
    
    st.markdown("### 📝 사용 방법")
    st.markdown("""
    1. 리서치할 주제를 입력하세요
    2. '보고서 생성' 버튼을 클릭하세요
    3. AI가 검색하고 분석한 리포트를 확인하세요
    """)
    
    st.divider()
    
    st.markdown("### 🔧 시스템 정보")
    st.markdown("""
    - **LLM**: Ollama (llama3.1)
    - **검색**: Tavily API
    - **검색 결과**: 최대 3개
    """)

# 메인 영역: 주제 입력 및 보고서 생성
col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_input(
        "🔍 리서치 주제를 입력하세요",
        placeholder="예: 트랜스포머 모델의 발전사",
        help="관심 있는 기술 주제나 트렌드를 입력하세요"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # 정렬을 위한 여백
    generate_button = st.button(
        "📄 보고서 생성",
        type="primary",
        use_container_width=True
    )

# 보고서 생성 로직
if generate_button:
    if not api_key or not validate_api_key(api_key):
        st.error("❌ Tavily API Key를 먼저 입력해주세요!")
    elif not topic:
        st.error("❌ 리서치 주제를 입력해주세요!")
    else:
        try:
            # 상태 표시와 함께 보고서 생성
            with st.status("🔄 AI 리포트 생성 중...", expanded=True) as status:
                st.write("🔍 검색 중...")
                st.caption(f"주제: {topic}")
                
                # 보고서 생성
                result = generate_report(topic)
                # 세션에 결과 저장 (재실행 시에도 유지)
                st.session_state["report_data"] = {
                    "report": result.get("report", ""),
                    "topic": topic,
                    "sources": result.get("sources", [])
                }
                
                st.write("✍️ 리포트 작성 중...")
                st.caption("검색 결과를 분석하고 구조화된 리포트를 생성하고 있습니다...")
                
                status.update(label="✅ 보고서 생성 완료!", state="complete", expanded=False)
            
            # 결과 출력
            st.success("🎉 리포트가 성공적으로 생성되었습니다!")
            
            # 보고서 본문 출력
            st.markdown("---")
            st.markdown(result["report"])
            
            # 참고 문헌 출력
            if result["sources"]:
                st.markdown("---")
                st.markdown("### 📚 참고 문헌")
                for idx, url in enumerate(result["sources"], 1):
                    st.markdown(f"{idx}. [{url}]({url})")
            
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.info("💡 다른 키워드나 주제로 다시 시도해보세요.")
            
        except ConnectionError as e:
            st.error(f"❌ {str(e)}")
            with st.expander("🔧 Ollama 실행 방법"):
                st.markdown("""
                1. 터미널에서 `ollama serve` 실행
                2. 모델 다운로드: `ollama pull llama3.1`
                3. 서버가 실행 중인지 확인: `ollama list`
                """)
                
        except Exception as e:
            st.error(f"❌ 오류가 발생했습니다: {str(e)}")
            
            # 일반적인 오류 해결 방법 안내
            with st.expander("🔧 문제 해결 방법"):
                st.markdown("""
                **Ollama 서버 오류:**
                - Ollama가 실행 중인지 확인하세요
                - 터미널에서 `ollama serve` 실행
                - `ollama pull llama3.1` 로 모델 다운로드 확인
                
                **API Key 오류:**
                - Tavily API Key가 올바른지 확인하세요
                - https://tavily.com 에서 새 키를 발급받으세요
                
                **검색 결과 없음:**
                - 다른 키워드나 주제로 다시 시도해보세요
                - 인터넷 연결 상태를 확인하세요
                """)

# 저장/다운로드 섹션 (세션에 결과가 있을 때 항상 표시)
if st.session_state["report_data"]:
    rd = st.session_state["report_data"]
    st.markdown("---")
    st.subheader("💾 리포트 저장 및 다운로드")

    # 파일명 입력
    default_name = ("".join(c for c in rd["topic"] if c.isalnum() or c in (" ", "_"))).strip().replace(" ", "_")[:50] or "report"
    filename = st.text_input("파일명(확장자 제외)", value=default_name, help="영문/숫자/언더스코어 권장")

    # 다운로드용 콘텐츠 구성 (참고 문헌 포함)
    content = rd["report"]
    if rd["sources"]:
        content += "\n\n---\n\n## 📚 참고 문헌\n\n" + "\n".join(f"{i+1}. {u}" for i, u in enumerate(rd["sources"]))
    st.download_button(
        label="Markdown 다운로드",
        data=content,
        file_name=f"{filename}.md",
        mime="text/markdown",
        key="download_md"
    )

# 푸터
st.divider()
st.caption("💡 Tip: 구체적인 주제일수록 더 상세한 리포트를 생성할 수 있습니다.")
