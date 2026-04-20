import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# ─── Verifica o dispositivo CUDA ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ─── Carregar ─────────────────────────────────────────────────────────────────────────
digits = load_digits()
X = digits.data
y = digits.target

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for digit, ax in enumerate(axes.flat):
    idx = np.random.choice(np.where(y == digit)[0])
    ax.imshow(digits.images[idx], cmap='gray')
    ax.set_title(f"Label: {y[idx]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# ─── Dividir em treino, teste e validação ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# ─── Normalização ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# ─── Converter para Tensores ─────────────────────────────────────────────────────────
# Importante: Envia os tensores para o dispositivo (CPU ou GPU) após a conversão
X_train     = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test      = torch.tensor(X_test, dtype=torch.float32).to(device)
X_val       = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train     = torch.tensor(y_train, dtype=torch.long).to(device)
y_test      = torch.tensor(y_test, dtype=torch.long).to(device)
y_val       = torch.tensor(y_val, dtype=torch.long).to(device)

# ─── Criando os minibatchs ──────────────────────────────────────────────────────────
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)

# ─── Criar o modelo para classificação ─────────────────────────────────────────────
class DigitsNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super(DigitsNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, 128)   # Entrada -> Oculta 128
        self.hidden2 = nn.Linear(128,64)            # Oculta 128 -> Oculta 64
        self.output = nn.Linear(64, n_classes)      # Oculta 64 -> Saída (classes)
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))          
        x = self.output(x)
        return x

# ─── Criar a instância do modelo, critério de perda e otimizador ─────────────────────
n_classes = len(torch.unique(y_train))
model = DigitsNet(input_size=X_train.shape[1], n_classes=n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── Treinamento ───────────────────────────────────────────────────────────────────
epochs              = 200
patience            = 15
best_val_loss       = np.inf
best_weights        = None

train_loss          = []
val_loss            = []
stop_epoch          = epochs
epochs_no_improve   = 0

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

    # Early Stopping
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
plt.show()

# ─── Avaliação do modelo no conjunto de teste ─────────────────────────────────────────
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.argmax(outputs, dim=1).cpu().numpy()

accuracy = accuracy_score(y_test.cpu().numpy(), predicted)
cmatrix = confusion_matrix(y_test.cpu().numpy(), predicted)
print(f'\nAcurácia Final: {accuracy:.4f}')
print("Matriz de Confusão:")
print(cmatrix)