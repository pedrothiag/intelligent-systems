import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

# Carregando o dataset
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)

# Divisão em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertendo para tensores do Pytorch
X_train = torch.tensor(X_train, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32)

# Criando os minibatchs
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)

# Cria o modelo para a regressão
class MLPRegression(nn.Module):
    def __init__(self, input_size):
        super(MLPRegression, self).__init__()
        self.hidden1 = nn.Linear(input_size, 64)  # Entrada -> Oculta 64
        self.hidden2 = nn.Linear(64, 32)          # Oculta 64 -> Oculta 32
        self.output = nn.Linear(32, 1)            # Oculta 32 -> Saída
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x

# Criar a instância do modelo
model = MLPRegression(X_train.shape[1])

# Parâmetros de treinamento
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Laco de treinamento
epochs = 50
train_loss = np.zeros(epochs)
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    # Apresenta os minibatches
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # Fase Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Fase Backward
        loss.backward()

        # Atualiza os pesos
        optimizer.step()
        epoch_loss += loss.item()
    
    # Apresenta o vetor de erros
    train_loss[epoch] = epoch_loss / len(train_loader)

# Avaliação do modelo
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    r2 = r2_score(y_test.numpy(), predictions.numpy())

print(f"\nMSE Final no Teste: {test_loss.item():.4f}")
print(f"R2: {r2:.4f}")

plt.figure()
epoch_vector = np.arange(1, epochs+1)
plt.plot(epoch_vector, train_loss)
plt.xlim([1, epochs])
plt.xlabel("Época de treinamento")
plt.ylabel("MSE")
plt.grid()
plt.show()