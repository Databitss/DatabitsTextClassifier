import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F

class LoaderData:
    def __init__(self, dataset, batch=32, seq=40, vocab=None, tokenizer=None, num_classes=10, j="train"):
        super().__init__()
        if j != "inference":
            print(f"Loading {j} data ...")
        self.batch, self.j = batch, j
        self.seq, self.vocab, self.tok = seq, vocab, tokenizer
        self.padding_idx = vocab["<PAD>"]
        self.num_classes = num_classes
        self.dataset = dataset

    def get_data(self):
        shuffle = True if self.j == "train" else False
        loader = DataLoader(self.dataset, batch_size=self.batch, shuffle=shuffle, collate_fn=self.pad_batch)
        return loader 

    def text_transform(self, x): 
        X = [self.vocab['[CLS]']] + [self.vocab['<BOS>']] + [self.vocab[token] for token in self.tok(x)] + [self.vocab['<EOS>']]
        return X
    
    def label_transform(self, x):
        return x-1

    def pad_batch(self, batch):
        if self.j != "inference":
            labels = [self.label_transform(z[0]) for z in batch]
        texts = [torch.tensor(self.text_transform(z[1]), dtype=torch.int64) for z in batch]
        padded_texts = []
        for text in texts:
            if text.size(0) > self.seq:
                text = text[:self.seq]
            else:
                text = F.pad(text, (0, self.seq - text.size(0)), value=self.padding_idx)
            padded_texts.append(text)
        x = torch.stack(padded_texts)
        if self.j != "inference":
            y = torch.tensor(labels, dtype=torch.int64)
            return x, y
        else:
            return x

    def data_for_inference(self):
        x = self.text_transform(self.dataset)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.tensor(x).to(device)
        if len(x) > self.seq:
            x = x[:self.seq]
        else:
            x = F.pad(x, (0, self.seq - len(x)), value=self.padding_idx)
        return x.unsqueeze(0)


