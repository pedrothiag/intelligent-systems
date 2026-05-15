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

# ─── Carregar ─────────────────────────────────────────────────────────────────────────
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)

# ─── Dividir em treino, teste e validação ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# ─── Normalização ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ─── Converter para Tensores ─────────────────────────────────────────────────────────
X_train = torch.tensor(X_train, dtype = torch.float32)
X_val = torch.tensor(X_val, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)
y_val = torch.tensor(y_val, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32)

# ─── Criando os minibatchs ──────────────────────────────────────────────────────────
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)

# ─── Criar o modelo para regressão ──────────────────────────────────────────────────
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

# ─── Criar a instância do modelo, critério de perda e otimizador ─────────────────────
model = MLPRegression(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── Treinamento ───────────────────────────────────────────────────────────────────
epochs = 200
patience = 15
best_val_loss = np.inf
best_weights = None

train_loss = []
val_loss = []
stop_epoch = epochs
epochs_no_improve = 0

# Laço de treinamento
for epoch in range(epochs):

    # Treinamento
    model.train()
    epoch_train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_loss.append(epoch_train_loss / len(train_loader))

    # Validação
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss.append(criterion(val_predictions, y_val).item())
    
    # Parada antecipada
    if val_loss[-1] < best_val_loss:
        best_val_loss = val_loss[-1]
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            stop_epoch = epoch + 1
            print(f"Early stopping at epoch {stop_epoch}")
            break

# Restaura os melhores pesos encontrados durante o treinamento
model.load_state_dict(best_weights)

# ─── Plotar a curva de perda de treinamento e validação ─────────────────────────────────
epoch_vector = np.arange(1, len(train_loss)+1)
plt.figure()
plt.semilogy(epoch_vector, train_loss, 'r', label = "Treinamento")
plt.semilogy(epoch_vector, val_loss, 'b', label = "Validação")
plt.xlim([1, len(train_loss)])
plt.xlabel("Época de treinamento")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# ─── Avaliação do modelo no conjunto de teste ─────────────────────────────────────────
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
r2 = r2_score(y_test.numpy(), predictions.numpy())

print(f"\nMSE Final no Teste: {test_loss.item():.4f}")
print(f"R2: {r2:.4f}")