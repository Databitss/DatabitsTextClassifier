Text Classifier using LSTM, GRU, and Transformer BERT

# Install Packgae
```python
!pip install databits
```

# Data preparation
Prepare the data X_train, y_train and X_test, y_test in list form. \
X_train -> list (text) \
X_test -> list (text) \
y_train -> lits label (integer starts from 1) \
y_test -> lits label (integer starts from 1) \

# Define Hyperparameters
```python
import torch
import torch.nn as nn
import numpy as np
from databits import CreateModel
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

BATCH_SIZE = 32
SEQUENCE_LENGTH = 100
EPOCHS = 5
EMBED_DIM = 512
N_LAYERS = 2
DROPOUT_RATE = 0.1
NUM_CLASSES = np.unique(np.array(y_train))
OPTIMIZER = torch.optim.Adam
LR = 0.001
LOSS = nn.CrossEntropyLoss
```

# Define Model
```python
model = CreateModel(X_train, y_train,
                 X_test, y_test,
                 batch=BATCH_SIZE,
                 seq=SEQUENCE_LENGTH,
                 embedding_dim=EMBED_DIM,
                 n_layers=N_LAYERS,
                 dropout_rate=DROPOUT_RATE,
                 num_classes=NUM_CLASSES)
```
# Train Model
```python
model.LSTM() # lstm model
model.GRU() # gru model
model.TRANFORMER() # tranformer model
model.BERT() # bert model
```
example, use gru model:
```python
model.GRU()
history = model.fit(epochs=EPOCHS, optimizer=OPTIMIZER, lr=LR, loss=LOSS)
```
example, use bert model:
```python
model.BERT()
history = model.fit(epochs=EPOCHS, optimizer=OPTIMIZER, lr=LR, loss=LOSS)
```

# Get y_true and predict label
```python
y_true, y_pred = model.eval() # no argumen needed
```

# Compute Accuracy, Precisiom, Recall, F1, and Cofusion Matrix
```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
accuracy = accuracy_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Akurasi: {accuracy:.4f}")

cm = confusion_matrix(y_true, y_pred)
print(cm)
```

# Inference
```python
text = "this is text"
pred = model.predict(text) # or
pred = model(text)
print(pred) # text label in int format
```
