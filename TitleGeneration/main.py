from lang_model import generate_title
from transformers import BertTokenizer
import torch
from fastapi import FastAPI
from typing import Union

global tokenizer
tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')

global model_gen
model_gen_re= torch.load('model/model_gen_re.pt', map_location = 'cpu')

app = FastAPI()


@app.get("/generate", summary='Simple fake news title generator (RU)', response_model=list[str])
def generate(starter: str = 'В России',  num: Union[int, None] = 1, model=model_gen_re, tokenizer=tokenizer):
    """
    Params:
    starter: a string to begin generation, required;
    num: number of sentences to generate, default = 1
    """
    return generate_title(model=model, tokenizer=tokenizer, starter=starter, num=num)


