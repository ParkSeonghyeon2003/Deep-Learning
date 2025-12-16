"""
LLM 모듈: Ollama를 사용한 로컬 LLM 초기화
"""
from langchain_ollama import ChatOllama
from logging_utils import llm as log_llm, success, info


def get_llm() -> ChatOllama:
    """
    로컬 Llama 3.1 모델을 초기화하여 반환합니다.
    
    Returns:
        ChatOllama: 초기화된 ChatOllama 인스턴스
    """
    model_name = "llama3.1"
    log_llm("LLM 초기화", kv={"model": model_name})
    llm = ChatOllama(
        model=model_name,
        temperature=0,
    )
    success("LLM 준비 완료", kv={"model": model_name})
    return llm
