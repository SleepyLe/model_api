from transformers import AutoTokenizer
from underthesea import word_tokenize
from clean_text import *
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import model


app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
model = model.SentimentClassifier()
model.load_state_dict(torch.load(f'content\last_step.pth',map_location=torch.device('cpu')))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Item(BaseModel):
    text: str


def clean_text(text):
    text = remove_url(text)
    text = handle_emoji(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = word_tokenize(text, format="text")
    return text

def format_lbl_1(lbl_prefix):
  lbl_prefix = lbl_prefix
  lbl_real=[]
  lbl_prefix = torch.where(torch.sigmoid(lbl_prefix) >= 0.5, torch.sigmoid(lbl_prefix), 0.)
  split_tensor = list(torch.tensor_split(lbl_prefix[0], 6))
  for i in range(len(split_tensor)):
    if(torch.count_nonzero(split_tensor[i]).item()==0):
      continue
    else:
      split_tensor[i] = torch.where(split_tensor[i] == torch.max(split_tensor[i]), 1, 0.)
  for i in range(len(split_tensor)):
    split_tensor[i] = split_tensor[i].tolist()
    if 1 in  split_tensor[i]:
      idx  = split_tensor[i].index(1)
      lbl_real.append(idx+1)
    else:
      lbl_real.append(0)
  return lbl_real


def infer(text, tokenizer, max_len=256):
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )   
    num=len(encoded_review)
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    test_1 = format_lbl_1(output)
    return test_1

@app.post("/items/")
async def create_item(item: Item):
    text = clean_text(item.text)
    res = infer(text,tokenizer)
    return res
