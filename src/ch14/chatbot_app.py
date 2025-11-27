import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from googletrans import Translator
import torch
import asyncio

async def translate_text(text, src, dest):
    translator = Translator()
    result = await translator.translate(text, src=src, dest=dest)
    return result.text

# 페이지 설정
st.set_page_config(
    page_title="Local LLM Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Local LLM Chatbot")
st.caption("SmolLM2-360M-Instruct 모델과 Googletrans를 활용한 한국어 챗봇")

# 모델 및 토크나이저 로드
@st.cache_resource
def load_model():
    # 예제에 명시된 체크포인트와 디바이스 설정 유지
    checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
    device = "cpu" # for CPU usage or "cuda" for GPU usage
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    
    return tokenizer, model, device

try:
    tokenizer, model, device = load_model()
except Exception as e:
    st.error(f"초기화 중 오류가 발생했습니다: {e}")
    st.stop()

# 세션 상태에 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 표시 및 저장
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 1. 한국어 -> 영어 번역
    try:
        translated_prompt = asyncio.run(translate_text(prompt, src='ko', dest='en'))
    except Exception as e:
        st.error(f"입력 번역 중 오류가 발생했습니다: {e}")
        translated_prompt = prompt

    # 모델 입력을 위한 메시지 구성
    # 예제와 동일하게 apply_chat_template 사용
    messages = [{"role": "user", "content": translated_prompt}]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # 2. 모델 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                outputs = model.generate(
                    inputs, 
                    max_new_tokens=100, # 답변 길이를 위해 50에서 100으로 조정 (필요시 수정 가능)
                    temperature=0.2, 
                    top_p=0.9, 
                    do_sample=True
                )
                
                # 생성된 텍스트 디코딩 (입력 프롬프트 제외)
                generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                
                # 3. 영어 -> 한국어 번역
                translated_response = asyncio.run(translate_text(generated_text, src='en', dest='ko'))
                
                st.markdown(translated_response)
                
                # 어시스턴트 응답 저장
                st.session_state.messages.append({"role": "assistant", "content": translated_response})
                
            except Exception as e:
                st.error(f"응답 생성 또는 번역 중 오류가 발생했습니다: {e}")
