import torch
from sentence_transformers import SentenceTransformer
from spacy.lang.ru import Russian
from torch.utils.data import Dataset
import torch.nn as nn


nlp = Russian()
nlp.add_pipe("sentencizer")
encoder =  SentenceTransformer('cointegrated/rubert-tiny2')

MAX_SEQ_LEN = 50  # обрезаем тексты статей до 50 предложений

device = 'cpu'
class CustomDataset(Dataset):
    pad = torch.zeros(312)

    def __init__(self, texts):
        self.texts = texts
        self.encoder = encoder

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        texts = self.texts[idx]

        sents = [sent.text for sent in nlp(texts).sents]

        if len(sents) > MAX_SEQ_LEN:
            sents = sents[:MAX_SEQ_LEN]

        encoding = torch.FloatTensor(self.encoder.encode(sents, show_progress_bar=False))

        if len(sents) < MAX_SEQ_LEN:  # padding
            pad = torch.zeros((MAX_SEQ_LEN - len(sents), 312))
            encoding = torch.cat([encoding, pad], dim=0)
        return {
            'text': texts,
            'input': encoding,

        }

HIDDEN_DIM = 300
N_LAYERS = 4
BIDIRECTIONAL = True
MAX_LEN = 312

class LSTMClassifier(nn.Module):

    def __init__(self, n_classes=9, embedding_dim=MAX_LEN, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                 bidirectional=BIDIRECTIONAL
                 ):
        super().__init__()

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=0.3,
            batch_first=True,
        )
        self.output_dim = n_classes
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, self.output_dim)

        self.func = nn.Tanh()

    def forward(self, inputs):
        outputs, (hidden, cell) = self.lstm(inputs)

        outputs = torch.mean(outputs, dim=1)

        outputs = self.fc(self.func(outputs))

        return outputs


def predict(model, test_loader):
    model.eval()
    model.to(device)
    with torch.no_grad():

        predicted_labels_list = []

        for data in test_loader:

            inputs = data['input'].to(device)

            outputs = model(inputs).to(device)

            predicted_labels = torch.argmax(outputs, dim=1)

            predicted_labels_list.extend(predicted_labels.tolist())

    decoded_labels = []
    for item in predicted_labels_list:
        if item == 5: res = 'Забота о себе/Здоровье'
        if item == 3: res = 'Страны бывшего СССР'
        if item == 2: res = 'Силовые структуры'
        if item == 8: res = 'Наука и техника'
        if item == 4: res = 'Спорт'
        if item == 7: res = 'Туризм/Путешествия'
        if item == 0: res = 'Общество'
        if item == 1: res = 'Экономика'
        if item == 6: res = 'Недвижимость/Строительство'
        decoded_labels.append(res)

    return decoded_labels