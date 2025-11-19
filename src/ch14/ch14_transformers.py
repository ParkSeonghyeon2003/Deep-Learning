#######################################################################
# sentiment-analysis (감성분석)

from transformers import pipeline

pipe = pipeline('sentiment-analysis', 
                model='nlptown/bert-base-multilingual-uncased-sentiment')

out = pipe('I am very happy')
out
out = pipe('I am very tired')
out
out = pipe('I am very sad')
out

#######################################################################
# text-generation (문장생성)

from transformers import pipeline

pipe = pipeline('text-generation')
# default model : openai-community--gpt2

out = pipe("Today is a beautiful day and I am feeling", 
           max_length = 30,         # 문장의 길이
           num_return_sequences=3)  # 생성할 문장 수
out
out[2]['generated_text']


#######################################################################
# text2text-generation

from transformers import pipeline

pipe = pipeline("text2text-generation")
# default model : google-t5--t5-base

out = pipe("translate from English to French: I'm very happy")
out
out = pipe("question: where is the capital of south korea?")
out
out = pipe("question: who is taller? context: John's height is 170 and Tom's height is 180")
out

#######################################################################
# question-answering

from transformers import pipeline
pipe = pipeline("question-answering")
# default model :distilbert-base-cased-distilled-squad
ctx = "My name is Ganesh and I am studying Data Science"

que = "What is Ganesh studying?"
out = pipe(context = ctx, question = que)
out
out['answer']
que = "Ganesh lieves in Korea?"
out = pipe(context = ctx, question = que)
out
out['answer']
