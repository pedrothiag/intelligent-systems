import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carregar e Normalizar
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converter para Tensores (PyTorch usa float32 para X e long para labels de classe)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.hidden1 = nn.Linear(4, 16)  # Entrada 4 -> Oculta 16
        self.output = nn.Linear(16, 3)   # Oculta 16 -> Saída 3 (classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.output(x)
        return x

model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50
train_loss = np.zeros(epochs)
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward
    loss.backward()
    
    # Update (Atualiza os pesos)
    optimizer.step()
    train_loss[epoch] = loss.item()

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test, predicted)
    cmatrix = confusion_matrix(y_test, predicted)
    print(f'\nAcurácia Final: {accuracy:.4f}')
    print(cmatrix)

epoch_vector = np.arange(1, epochs+1)
plt.figure()
plt.plot(epoch_vector, train_loss)
plt.xlabel("Época de treinamento")
plt.ylabel("Loss")
plt.grid()
plt.show()