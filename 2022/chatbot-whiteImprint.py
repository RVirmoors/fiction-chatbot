# select the type of architecture: GPT-2 or NEO
model_type = 'gpt2' # 'gpt2' or 'neo'
# choose the exact model name
model_name = 'gpt-combinedplus2'  # 'gpt2' or 'EleutherAI/gpt-neo-125M' or one of your finetuned folders: 'finetuned' etc
# add the length of the prompt tokens to match with the mesh-tf generation
max_length = 250
# number of generated texts
num_generate = 3

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
mem = ""

remember_QA = True
show_log = False # this is what the user will see, scrolling on the screen



# https://github.com/Xirider/finetune-gpt2xl
# credit to Niels Rogge - https://github.com/huggingface/transformers/issues/10704

import torch
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

print(device)

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# listens for changes in q.txt, writes answer to log.txt
class MyHandler(FileSystemEventHandler):
  def on_any_event(self, event):
    print(event.event_type)
    global model_type, model_name, max_length, num_generate, head, remember_QA, show_log, device, prompt, mem, tokenizer, model
    print(f'event type: {event.event_type}  path : {event.src_path}')
    if event.src_path == "C:\\Users\\ITPMA - master\\Desktop\\q.txt":

      with open("C:\\Users\\ITPMA - master\\Desktop\\q.txt") as f:
        print("Old question:", prompt)
        new_prompt = f.read()
        # strip whitespace before punctuation
        new_prompt = re.sub(r'\s([?.!"](?:\s|$))', r'\1', new_prompt)
        if new_prompt == prompt:
          return
        prompt = new_prompt
        print("New question:", prompt)

      log = ""
      with open("log.txt") as f:
        log = f.read()

      log = log + "\nQ: " + prompt + "\n"
      
      if remember_QA:
        text = head + mem + "Q: " + prompt + "\n"
      else:
        text = head + "Q: " + prompt + "\n"
      
      mem = "\nQ: " + prompt + "\n"

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
          print(">> GENERATE >>", time.time() - start)
          generated_texts = tokenizer.batch_decode(
              generated_ids, skip_special_tokens=True)

        if model_type == 'neo':
            start = time.time()
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            print(">> TOKENIZER >>", time.time() - start)
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
            print(">> GENERATE >>", time.time() - start)
            generated_texts = tokenizer.batch_decode(
                gen_tokens, skip_special_tokens=True)
            print(">> DECODE >>", time.time() - start)
        #print(generated_texts)
        #print(text)
        
        counter = 1
        for generated_text in generated_texts:
          print(generated_text.split(text))
          generated_text = generated_text.split(text)[1]
        
          answer = generated_text
          # split on all punctuation, take just first after `A:`
          # https://bobbyhadz.com/blog/python-split-string-on-punctuation
          #answer = re.split( r'()[.!?]', answer)[0]
          #https://stackoverflow.com/questions/40736948/regex-string-repetition-of-min-length
          answer = re.sub(r"(.{2,}?)\1+", r"\1", answer)
          answer = answer.split("Q:")[0]
          answer = answer.split("\n")[0]
          #answer = answer + '.'

          # remove anything at the end without punctuation
          tail = answer.split(".")[-1]
          print("tail", tail, "len", len(tail))
          if len(tail) > 2:
            answer = answer.split(tail)[0]

          print("===",counter, "===\n", generated_text)
          counter = counter + 1

          no_words = len(answer.split())
          if no_words >= 6:
            break
        print("PROCESSED ANSWER:\n", answer)

      log = log + answer + "\n"
      mem = mem + answer + "\n"

      if show_log:
        print("LOG (USER SEES THIS):\n", log)
      with open('log.txt', 'w') as f:
        f.write(log)

      #tokenizer, model = loadModel()


def loadModel():
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
    model = GPTNeoForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(device) # .half() # , torch_dtype=torch.float32

  if model:
    print("model loaded:", model_type, "/", model_name)
    print(">> MODEL >>", time.time() - start)

  return tokenizer, model

if __name__ == "__main__":
  tokenizer, model = loadModel()
  event_handler = MyHandler()
  observer = Observer()
  observer.schedule(event_handler, path='C:\\Users\\ITPMA - master\\Desktop\\', recursive=False)
  observer.start()

  try:
      while True:
          time.sleep(1)
  except KeyboardInterrupt:
      observer.stop()
  observer.join()

# C:\\Users\\ITPMA - master\\Desktop\\