# SmolLM
########################################################
# code 1 . HF example
########################################################

from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cpu" # for CPU usage or "cuda" for GPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

messages = [{"role": "user", "content": "What is the capital of France."}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, 
                         max_new_tokens=50, 
                         temperature=0.2, 
                         top_p=0.9, 
                         do_sample=True)
print(tokenizer.decode(outputs[0]))


messages = [{"role": "user", 
             "content": "Show me python code for histogram chart"}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, 
                         max_new_tokens=50, 
                         temperature=0.2, 
                         top_p=0.9, 
                         do_sample=True)
print(tokenizer.decode(outputs[0]))


##################################################################
# Code 2
##################################################################

from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cpu" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

def get_answer(messages):
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    #print(input_text)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, 
                            max_new_tokens=1024, 
                            temperature=0.2, 
                            top_p=0.9, 
                            do_sample=True)
    
    tmp = tokenizer.decode(outputs[0])
    tmp = tmp[-(len(tmp)-len(input_text)):]

    tmp = tmp.replace("<|im_start|>assistant\n", "")
    tmp = tmp.replace("<|im_end|>", "")
        
    return(tmp)

messages = [{"role": "user", "content": "What is the capital of France."}]
#messages = [{"role": "user", "content": "Show me python code for histogram chart"}]
answer = get_answer(messages)
print(answer)

#########################################################################
# code 3
#########################################################################
from transformers import pipeline

generator = pipeline(task="text-generation", 
                     model=checkpoint, 
                     tokenizer=checkpoint, 
                     truncation=True,
                     device='cpu')

chat = [
    {"role": "system", "content": "You are a kind assistant."},
    {"role": "user", "content": "Show me python code for histogram chart"}
]
chat = [
    {"role": "system", "content": "You are a kind assistant."},
    {"role": "user", "content": "How may cars are eaten by human?"}
]


result = generator(chat, 
                   max_length=600, 
                   do_sample=True, 
                   temperature=0.2, 
                   top_p=0.9)

print(result[0]['generated_text'][2]['content'])


