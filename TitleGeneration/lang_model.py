import torch
import torch.nn as nn
import random
import numpy as np


VOCAB_SIZE = 83828
MAX_LEN = 2048
HIDDEN_DIM = 512
N_LAYERS = 2
BIDIRECTIONAL = False


class LSTMClassifier(nn.Module):

    def __init__(self, n_classes=VOCAB_SIZE, vocab_size=VOCAB_SIZE, embedding_dim=MAX_LEN, hidden_dim=HIDDEN_DIM,
                 n_layers=N_LAYERS, bidirectional=BIDIRECTIONAL
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=0.3,
            batch_first=True,
        )
        self.hidden_dim = hidden_dim
        self.output_dim = n_classes
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.projection = nn.Linear(self.hidden_dim, self.output_dim)
        self.func = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        inputs = self.embedding(inputs)

        outputs, (hidden, cell) = self.rnn(inputs)
        outputs = self.dropout(self.linear(self.func(outputs)))
        projection = self.projection(self.func(outputs))

        return projection


def generate_sequence(model, starter, tokenizer, max_seq_len=20) -> str:
    model = model.to('cpu')
    input_ids = tokenizer.encode_plus(starter)['input_ids']
    input_ids.pop(-1)
    sentence = torch.LongTensor(input_ids).to('cpu')
    model.eval()
    with torch.no_grad():
        for i in range(max_seq_len - len(input_ids)):
            next_word_distribution = model(sentence)[-1]
            next_word_sample = torch.topk(next_word_distribution, 5)[1].detach().cpu().numpy()
            next_word = torch.LongTensor(np.random.choice(next_word_sample, 1)).to('cpu')
            sentence = torch.cat([sentence, next_word])

            if next_word.item() == 3:
                break

    sequence = tokenizer.decode(sentence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return sequence


class Node:
    def __init__(self, sequence, prob=1.0, idx=None):
        self.sequence = sequence
        self.prob = prob
        self.idx = idx
        self.childs = []

    def __repr__(self):
        return (
                f"Node\n sequence: {self.sequence.tolist()}, "
                + f"index: {self.idx}, prob: {self.prob}, "
                + f"num childs: {len(self.childs)}"
        )


def beam_search(
        model: torch.nn.Module, parents: list[Node], bins: int = 15, k: int = 8, eos_id: int = 3) -> torch.Tensor:
    """
    Generate sequence via beam search algorithm.
    Parameters
    ----------
    model : torch.nn.Module
        Trained pytorch model.
    parents : list[Node]
        List of parent nodes.
    bins : int, default = 8
        Maximum number of hypothesis.
    k : int, default = 2
        Number of searches.
    eos_id : int, default = 3
        End of sentence (<eos>) token id.
    Returns
    -------
    torch.Tensor of int's (1, N)
        Most probable hypothesis.
    """

    model.eval()
    model.to('cpu')
    childs = []
    for parent in parents:
        if parent.idx == eos_id:
            childs.append(parent)
        else:
            dist = model.forward(parent.sequence)[:, -1]
            probs, ids = torch.topk(torch.softmax(dist, dim=-1), k=k, dim=-1)
            probs /= torch.sum(probs)
            for prob, idx in zip(probs.flatten(), ids.flatten()):
                node = Node(
                    sequence=torch.cat((parent.sequence, idx.view(-1, 1)), 1),
                    idx=idx,
                    prob=parent.prob * prob,
                    )
                # parent.childs.append(node) # uncomment to debug root node
                childs.append(node)

    if len(childs) > bins:
        childs, _ = zip(
            *sorted([(node, 1.0 - node.prob) for node in childs], key=lambda x: x[1])[:bins]
            )  # keep bins with the highest prob
    # Terminate if <eos> appear in all sequences
    if all([node.idx == eos_id for node in childs]):
        priority = random.choice(np.argpartition([node.prob for node in childs], 10))
        return childs[priority].sequence

    return beam_search(model, childs, bins, k, eos_id)


def generate_title(model, tokenizer, starter, num):
    model.eval()
    model.to('cpu')
    sentences = []
    with torch.no_grad():
        for i in range(num):
            starter_ext = generate_sequence(model, starter, tokenizer, max_seq_len=4)
            input_ids = tokenizer.encode_plus(starter_ext)['input_ids']
            input_ids.pop(-1)
            ids = torch.LongTensor([input_ids]).to('cpu')

            root = Node(ids)
            seq = beam_search(model, [root]).flatten()
            sentences.append(tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return sentences
