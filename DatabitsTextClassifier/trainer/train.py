import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from collections import Counter
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math

from DatabitsTextClassifier.model import TransformerEncoder, BERT, GRUModel, LSTMModel
from DatabitsTextClassifier.loader import LoaderData

class CreateModel(nn.Module):
    def __init__(self,
                 X_train, y_train,
                 X_test, y_test,
                 batch=32,
                 seq=40,
                 embedding_dim=512,
                 n_layers=2,
                 dropout_rate=0.3,
                 num_classes=10):
        super().__init__()
        print("Loading setup data ...")
        torch.cuda.empty_cache()
        self.train_data = list(zip(y_train, X_train))
        self.test_data = list(zip(y_test, X_test))
        self.batch, self.seq = batch, seq
        self.embed = embedding_dim
        self.n_layers = n_layers
        
        self.tokenizer = get_tokenizer('basic_english')
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield self.tokenizer(text)
        self.vocab = build_vocab_from_iterator(
            yield_tokens(self.train_data + self.test_data),
            specials=['<unk>', '<BOS>', '<EOS>', '<PAD>', '[CLS]']
        
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        
        self.loader_train = LoaderData(self.train_data, self.batch, self.seq, self.vocab, 
                                       self.tokenizer, num_classes, j="train")
        self.loader_test = LoaderData(self.test_data, self.batch, self.seq, self.vocab, 
                                      self.tokenizer, num_classes, j="val")
        self.train_loader= self.loader_train.get_data()
        self.val_loader = self.loader_test.get_data()
        
        self.num_cls = num_classes
        self.drp = dropout_rate
        self.model = None
        self.bert = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocab_size = len(self.vocab)
        print("Successful load model")

    def forward(self, x):
        self.load = LoaderData(x, self.batch, self.seq, self.vocab, 
                               self.tokenizer, self.num_cls, j="inference")
        data = self.load.data_for_inference()
        out = self.model(data)
        _, pred = torch.max(out, dim=1)
        return pred

    def predict(self, x):
        return self.forward(x)
        
    def LSTM(self):
        torch.cuda.empty_cache()
        self.model = LSTMModel(self.vocab_size, self.embed, self.num_cls, self.n_layers, dropout=self.drp)
        self.model.to(self.device)
        self.bert = False
        return self.model

    def GRU(self):
        torch.cuda.empty_cache()
        self.model = GRUModel(self.vocab_size, self.embed, self.num_cls, self.n_layers, dropout=self.drp)
        self.model.to(self.device)
        self.bert = False
        return self.model

    def TRANSFORMER(self, num_heads=8):
        torch.cuda.empty_cache()
        self.model = TransformerEncoder(embed_dim=self.embed, num_heads=num_heads, sequence_length=self.seq, 
                              vocab_size=self.vocab_size, n_layers=self.n_layers, bert=False, num_cls=self.num_cls)
        self.model.to(self.device)
        self.bert = False
        return self.model

    def BERT(self, num_heads=8):
        torch.cuda.empty_cache()
        self.model = BERT(embed_dim=self.embed, num_heads=num_heads, sequence_length=self.seq, 
                              vocab_size=self.vocab_size, n_layers=self.n_layers, bert=True, num_cls=self.num_cls)
        self.model.to(self.device)
        self.bert = True
        return self.model

    def train(self, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        with tqdm(self.train_loader, desc="Training", unit="batch") as t:
            for X, y in t:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, pred = torch.max(out, dim=1)
                correct += (pred == y).sum().item()
                total_samples += y.size(0)
                # t.set_postfix(loss=loss.item(), accuracy=correct / total_samples)
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total_samples
        return avg_loss, accuracy

    def eval(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        all_labels, all_preds = [], []
        with tqdm(self.val_loader, desc="Validation", unit="batch") as t:
            with torch.no_grad():
                for X, y in t:
                    X, y = X.to(self.device), y.to(self.device)
                    out = self.model(X)
                    _, pred = torch.max(out, dim=1)
                    all_labels.extend(y.cpu().numpy())
                    all_preds.extend(pred.cpu().numpy())
        return all_labels, all_preds

    def validate(self, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        with tqdm(self.val_loader, desc="Validation", unit="batch") as t:
            with torch.no_grad():
                for X, y in t:
                    X, y = X.to(self.device), y.to(self.device)
                    out = self.model(X)
                    loss = criterion(out, y)
                    total_loss += loss.item()
                    _, pred = torch.max(out, dim=1)
                    correct += (pred == y).sum().item()
                    total_samples += y.size(0)
                    # t.set_postfix(loss=loss.item(), accuracy=correct / total_samples)
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total_samples
        return avg_loss, accuracy

    def fit(self, epochs, optimizer, lr, loss, loss_w=None):
        optimizer, criterion = self.get_optim(optimizer, lr, loss, loss_w)
        t_acc, t_loss, v_acc, v_loss = [], [], [], []
        best_val_loss = float('inf')
        best_model_state_dict = None
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train(optimizer, criterion)
            val_loss, val_accuracy = self.validate(criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = self.model.state_dict()
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}\n")
            t_loss.append(train_loss)
            t_acc.append(train_acc)
            v_loss.append(val_loss)
            v_acc.append(val_accuracy)
        if best_model_state_dict:
            self.model.load_state_dict(best_model_state_dict)
            print("Restored model to the best state based on validation loss.")
        history = {"Train Loss":t_loss, "Train Acc":t_acc, "Val Loss":v_loss, "Val Acc":v_acc}
        return history
        
    def get_optim(self, optimizer, lr, loss, loss_w):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        if loss_w is not None:
            try:
              criterion = loss(weight=loss_w)
            except:
              criterion = loss(pos_weight=loss_w)
        else:
            criterion = loss()
        return optimizer, criterion
