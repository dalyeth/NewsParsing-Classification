from classifier import CustomDataset, predict
import torch
from torch.utils.data import DataLoader
import pandas as pd
from classifier import LSTMClassifier
from fastapi import FastAPI, UploadFile, File


demo = pd.read_csv('demo/demo.csv')
demo_ds = CustomDataset(list(demo['text']))
demo_dl = DataLoader(demo_ds, batch_size=1)

global model_class
model_class = torch.load('model/model_class_re.pt', map_location ='cpu')


app = FastAPI()


@app.post("/upload", summary = 'upload a csv file with news articles and classify ')
def upload(column_name: str,  file: UploadFile = File(...)):
    """
    :param column_name: name of the column which contains the article text
    :param file: csv file with news articles
    :return: {id: category }
    """

    data = pd.read_csv(file.file)
    file.file.close()
    ds = CustomDataset(list(data[column_name]))
    dl = DataLoader(ds, batch_size=1)

    data['res']=predict(model_class, dl)

    return  data['res']