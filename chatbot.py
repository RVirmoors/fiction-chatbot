# select the type of architecture: GPT-2 or NEO
model_type = 'gpt2' # 'gpt2' or 'neo'
# choose the exact model name
model_name = 'gpt2'  # 'gpt2' or 'EleutherAI/gpt-neo-125M' or one of your finetuned folders: 'finetuned' etc
# add the length of the prompt tokens to match with the mesh-tf generation
max_length = 250
# number of generated texts
num_generate = 3

remember_QA = True
show_log = True # to display the whole history to the user
show_debug = False # display unprocessed / raw generated text


import re
head = """[An oracle from the future.]

Q: Who are you?
A: My identity is of no importance.
Q: Who am I?
A: You are a human with a thirst for knowledge. Ask of me all the questions you desire.
Q: What is my future?
A: The future is where you are going to spend the rest of your life. Future events such as these will affect you in the future. 
"""

prompt = ""



# https://github.com/Xirider/finetune-gpt2xl
# credit to Niels Rogge - https://github.com/huggingface/transformers/issues/10704

import time
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loadModel():
  global device
  start = time.time()
  if model_type == 'gpt2':
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(device)

  if model_type == 'neo':
    from transformers import GPTNeoForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device == 'cpu':
      model = GPTNeoForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(device)
    else:
      model = GPTNeoForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).half().to("cuda")

  if model:
    print("model loaded:", model_type, "/", model_name)
    #print("in", time.time() - start)

  return tokenizer, model



def generate(tokenizer, model, log=""):
  global device, max_length, num_generate, head, remember_QA
  prompt = ""
  while len(prompt) == 0:
    prompt = input("Q: ")
    prompt = str(prompt)

  log = log + "Q: " + prompt + "\n"
    
  if remember_QA:
    text = head + log
  else:
    text = head + "Q: " + prompt + "\n"

  no_words = 0
  while no_words < 6:
  # the generated (and processed) answer has to have at least 5 words!
    if model_type == 'gpt2':
      start = time.time()
      encoding = tokenizer(text, padding=True, return_tensors='pt').to(device)
      max_length = max_length + len(encoding)
      with torch.no_grad():
          generated_ids = model.generate(
              **encoding,
              num_return_sequences=num_generate,
              do_sample=True,
              max_length=max_length,
              top_k=50, 
              top_p=0.95,
              use_cache=True
            )
      #print(">> GENERATE >>", time.time() - start)
      generated_texts = tokenizer.batch_decode(
          generated_ids, skip_special_tokens=True)

    if model_type == 'neo':
      start = time.time()
      ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
      #print(">> TOKENIZER >>", time.time() - start)
      max_length = max_length + ids.shape[1]
      gen_tokens = model.generate(
          ids,
          num_return_sequences=num_generate,
          do_sample=True,
          max_length=max_length,
          temperature=0.7,
          #top_k=50, 
          #top_p=0.95,
          use_cache=True
      )
      #print(">> GENERATE >>", time.time() - start)
      generated_texts = tokenizer.batch_decode(
          gen_tokens, skip_special_tokens=True)
      #print(">> DECODE >>", time.time() - start)
      
    counter = 1
    for generated_text in generated_texts:
      answer = processText(generated_text, text)
      if show_debug:
        print("===",counter, "===\n", generated_text)
      counter = counter + 1

      no_words = len(answer.split())
      if no_words >= 6:
        break
      
    if show_debug:
      print("PROCESSED ANSWER:\n", answer)
    else:
      print(answer)
    log = log + answer + "\n"
    return log


def processText(generated_text, text):
  # print(generated_text.split(text))
  generated_text = generated_text.split(text)[1]

  answer = generated_text
  # split on all punctuation, take just first after `A:`
  # https://bobbyhadz.com/blog/python-split-string-on-punctuation
  #answer = re.split( r'()[.!?]', answer)[0]
  #https://stackoverflow.com/questions/40736948/regex-string-repetition-of-min-length
  answer = re.sub(r"(.{4,}?)\1+", r"\1", answer)
  answer = answer.split("Q:")[0]
  answer = answer.split("\n")[0]
  #answer = answer + '.'

  # remove anything at the end without punctuation
  tail = answer.split(".")[-1]
  #print("tail", tail, "len", len(tail))
  if len(tail) > 2:
    answer = answer.split(tail)[0]
  return answer

if __name__ == "__main__":
  # load the model
  tokenizer, model = loadModel()
  # set up the log
  log = ""
  with open("log.txt") as f:
    log = f.read()
  # start dialogueing
  while True:
    log = generate(tokenizer, model, log)
    if show_log:
      print("\n== LOG ==\n", log)
    with open('log.txt', 'w') as f:
      f.write(log)