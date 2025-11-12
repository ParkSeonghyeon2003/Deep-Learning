import gensim
import urllib.request

# 구글의 사전 훈련된 Word2Vec 모델을 다운로드 (명령 프롬프트에서 실행)
# pip install gdown
# gdown https://drive.google.com/uc?id=1Av37IVBQAAntSe1X3MOAl5gvowQzd2_j

# 구글의 사전 훈련된 Word2Vec 모델을 로드.
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)

# 모델의 크기 출력
print(word2vec_model.vectors.shape)

# 두 단어의 유사도 계산
print(word2vec_model.similarity('this', 'is'))
print(word2vec_model.similarity('post', 'book'))

# 임베딩 벡터 출력
print(word2vec_model['book'])
