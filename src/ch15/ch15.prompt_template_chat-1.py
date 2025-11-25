# template for chat

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# prompt message
message = [
    ("system", "너는 여행관련 전문가야."),
    ("system", "너는 한국어로 대화해."),
    ("placeholder", "{chat_history}"),
    ("user", "{user_input}"),
]

# define chat prompt
chat_prompt = ChatPromptTemplate.from_messages(message)

# model
llm_model = ChatOllama(model="PetrosStav/gemma3-tools:4b")
# 체인 생성
rag_chain = chat_prompt | llm_model 

# chat history 초기화
chat_history = [
    ("ai", "안녕하세요, 저는 여행 전문가 브라이언 입니다. 반갑습니다."),
    ("user", "안녕, 나는 길동이야"),
    ("user", "나는 여행을 좋아해. 너는 어떤 여행지를 추천해줄 수 있어?"),
    ("ai", "세계 여러나라의 여행지를 추천해 줄 수 있습니다. 어느곳에 대해 알고 싶으신가요?"),
]

def answer_for(user_input):
    chat_history.append(
      ("human", user_input,)
    )
    whole_answer = ""
    for chunk in rag_chain.stream({"chat_history": chat_history,
                                   "user_input": user_input}):
        print(chunk.content, end="", flush=True)    
        whole_answer += chunk.content

    chat_history.append(
      ("ai", whole_answer,)
    )
    print("\n")


# chatting test #######################################
user_input = "안녕 너의 이름은 뭐야?"
answer_for(user_input)

user_input = "싱가포르 여행지 3곳 추천해줘"
answer_for(user_input)

user_input = "첫번째 여행지의 입장료는 얼마야?"
answer_for(user_input)
