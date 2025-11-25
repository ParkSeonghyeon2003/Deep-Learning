# Basic template for a prompt to be used with an LLM.

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# prompt template
template = """
     당신은 QA(Question-Answering)을 수행하는 Assistant입니다. 
    {question}에 대해 설명하세요 
    5문장 이내로 으로 답변하세요.
    """

# Added prompt template
prompt = ChatPromptTemplate.from_template(template)

# model
llm_model = ChatOllama(model="PetrosStav/gemma3-tools:4b")

# 체인 생성
rag_chain =  prompt | llm_model 

# 스트리밍 출력함수 정의
def answer_for(question):
    print("Answer :")
    for chunk in rag_chain.stream({"question": question}):
        print(chunk.content, end="", flush=True)    

    print("\n")

# Q&A 테스트 
question = "한국의 가을 날씨"
answer_for(question)

# 참고. 일괄응답 (출력시간 오래걸림)
# result = rag_chain.invoke({"question": question})
# print(result.content)