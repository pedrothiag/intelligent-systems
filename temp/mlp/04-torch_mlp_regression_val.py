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

# Divisão em conjunto de treinamento (70%), validação (15%) e teste (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Normalização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convertendo para tensores do Pytorch
X_train = torch.tensor(X_train, dtype = torch.float32)
X_val = torch.tensor(X_val, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)
y_val = torch.tensor(y_val, dtype = torch.float32)
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
val_loss = np.zeros(epochs)

for epoch in range(epochs):

    # ── Fase de treino ──────────────────────────────────────────
    model.train()
    epoch_train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_loss[epoch] = epoch_train_loss / len(train_loader)

    # ── Fase de validação ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss[epoch] = criterion(val_predictions, y_val).item()

# Avaliação do modelo
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    r2 = r2_score(y_test.numpy(), predictions.numpy())

print(f"\nMSE Final no Teste: {test_loss.item():.4f}")
print(f"R2: {r2:.4f}")

epoch_vector = np.arange(1, epochs+1)
plt.figure()
plt.plot(epoch_vector, train_loss, 'r', label = "Treinamento")
plt.plot(epoch_vector, val_loss, 'b', label = "Validação")
plt.xlim([1, epochs])
plt.xlabel("Época de treinamento")
plt.ylabel("MSE")
plt.grid()
plt.legend()
plt.show()