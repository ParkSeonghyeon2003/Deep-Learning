# few shot prompting
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# prompt template
template = """
     당신은 QA(Question-Answering)을 수행하는 Assistant입니다. 
    {question}에 대해 답변하세요.
    """

# Added prompt template
prompt = ChatPromptTemplate.from_template(template)

# model
llm_model = ChatOllama(model="PetrosStav/gemma3-tools:4b")

# 체인 생성
rag_chain =  prompt | llm_model 

# 스트리밍 출력함수 정의
def get_answer(question):
    print("Answer :")
    for chunk in rag_chain.stream({"question": question}):
        print(chunk.content, end="", flush=True)    

    print("\n")

# zero-shot prompting
Q1 = """
다음 문장에 포함된 감정이 긍정인지 부정인지 판단해줘:
'오늘 날씨가 좋아서 기분이 좋아요.'
"""
# one-shot prompting
Q2 = """
다음 문장에 포함된 감정이 긍정인지 부정인지 판단해줘:
'오늘 날씨가 좋아서 기분이 좋아요.'
다음 예시를 참고해서 판단해줘:
[문장] 딥러닝과목 A+ 받아서 신나요.
[감정] 긍정
"""
# few-shot prompting
Q3 = """
다음 문장에 포함된 감정이 긍정인지 부정인지 판단해줘:
'오늘 날씨가 좋아서 기분이 좋아요.'
다음 예시를 참고해서 판단해줘:
[문장] 딥러닝과목 A+ 받아서 신나요.
[감정] 긍정
[문장] 수업시간에 졸았어요. 망했네요.
[감정] 부정
[문장] 나는 지하철로 통학합니다.
[감정] 중립
"""

get_answer(Q1)
get_answer(Q2)
get_answer(Q3)

