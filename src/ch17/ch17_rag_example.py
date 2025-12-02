# (1) 문서 불러오기
from langchain_community.document_loaders import UnstructuredURLLoader
# import nltk

#nltk.download("punkt")   # 토큰화에 필요한 패키지 (관련 에러발생시 실행)
#nltk.download("averaged_perceptron_tagger")  # (관련 에러발생시 실행)

urls = [
      "https://m.blog.naver.com/bds6546/221150524290",
]

loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()

print(f"불러온 문서의 개수: {len(docs)}")
print(docs[0])

# (2) 문서 나누기 (Chuncking)   ##################################
# Text Split (Documents -> small chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                               chunk_overlap=20)
chunked_docs = text_splitter.split_documents(docs)

print(len(chunked_docs))
print(chunked_docs[10])


# (3) 벡터 데이터베이스에 임베딩 저장

# (3)-1 임베딩 생성
from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

embedding_model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    encode_kwargs={"normalize_embeddings": True},
)

# (3)-2 벡터 데이터베이스 생성
vectorstore = FAISS.from_documents(
    documents=chunked_docs, embedding=embedding_model,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
)

vector_count = vectorstore.index.ntotal
print(f"저장된 벡터의 개수: {vector_count}")


# 임베딩 테스트

user_query = "대한민국의 주권은 누구에게 있나요?"
query_vector = embedding_model.embed_query(user_query)

print(query_vector)
print(f"임베딩 차원 수: {len(query_vector)}")

# (4) 검색기(retriever) 만들기
# 벡터 저장소에서 기본 설정을 사용하여 검색기를 초기화하고,
# 주어진 쿼리에 대해 검색을 수행하는 예시

user_query = "정당 설립은 정부의 허가가 필요한가요?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
basic_docs = retriever.invoke(user_query)
for doc in basic_docs:
    print(doc.metadata, doc)

# (5) 프롬프트 준비하기
# Langchain 라이브러리를 사용하여 QA(Question-Answering) Assistant의
# 프롬프트를 생성하는 방법 구현

from langchain_core.prompts import ChatPromptTemplate

# prompt template
template = """
    <s>[INST] 당신은 대한민국 헌법 전문가로서 QA(Question-Answering)을 수행하는
    Assistant입니다. 다음의 Context를 이용하여 Question에 답변하세요.
    최소 3문장 최대 5문장으로 답변하세요.
    주어진 Context가 없다면 "정보가 부족하여 답변할 수 없습니다."를 출력하세요.
    Context: {context}
    Question: {question} [/INST]
    Answer:
    """

# Added prompt template
prompt = ChatPromptTemplate.from_template(template=template)

prompt.pretty_print()

from langchain_ollama import ChatOllama

# (6) Chain 구성하기

# model
llm_model = ChatOllama(model="PetrosStav/gemma3-tools:4b")

# 체인 생성
rag_chain = rag_chain = prompt | llm_model

# (7) Q&A 테스트

def answer_for(user_input):
    whole_answer = ""
    basic_docs = retriever.invoke(user_input)
    for chunk in rag_chain.stream({"context": basic_docs,
                                   "question": user_input}):
        print(chunk.content, end="", flush=True)
        whole_answer += chunk.content

    print("\n")


question = "정당 설립은 자유인가요?"
answer_for(question)
question = "근로자의 권리는 무엇인가요?"
answer_for(question)