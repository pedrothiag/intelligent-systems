import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Verifica se existe um dispositivo CUDA disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ─────── Classe para a CNN  ────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1,   out_channels=6,   kernel_size=5),   # → 6@28×28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                         # → 6@14×14
            nn.Conv2d(in_channels=6,   out_channels=16,  kernel_size=5),   # → 16@10×10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                         # →  16@5×5 = 16x5x5 = 400
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                                 # 16×5×5 = 400
            nn.Linear(400, 120), nn.ReLU(),                               # C5: 120
            nn.Linear(120, 84),  nn.ReLU(),                               # F6: 84
            nn.Linear(84, num_classes),                                   # OUTPUT: 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))

# ─────── Funcao Main  ───────────────────────────────────────
if __name__ == '__main__':
    # Cria o transfom
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    # Carregar o dataset (treino e teste)
    full_train_dataset = MNIST(root="data", train=True,  download=True, transform=transform)
    test_dataset       = MNIST(root="data", train=False, download=True, transform=transform)

    # Cria o conjunto de validação para a parada antecipada
    val_size   = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
    )

    # Cria os dataloaders de treinamento, validação e teste
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # Instancia o modelo e armazena na GPU
    model  = LeNet5().to(device)

    # Cria os parâmetros do otimizador
    optimizer     = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion     = nn.CrossEntropyLoss()
    epochs        = 50
    patience      = 5

    # Variaveis para a parada antecipada
    best_val_loss = float("inf")
    best_weights  = None
    patience_count = 0

   # ─────── Laço de Treinamento  ───────────────────────────────────────
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(1, epochs + 1):
        
        # Treinamento     
        model.train()
        train_loss, train_correct = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(X)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
        
        # Calcular o train_loss e o train_acc
        train_loss  = train_loss/len(train_dataset)
        train_acc   = train_correct/len(train_dataset)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validação
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits      = model(X)
                val_loss   += criterion(logits, y).item() * len(X)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
        
        # Calcular o val_loss e val_acc
        val_loss  = val_loss/len(val_dataset)
        val_acc   = val_correct / len(val_dataset)  
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print (verificar o processo de treinamento)
        print(f"Epoch {epoch:2d}/{epochs} "
              f"| train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
              f"| val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        # Earlying Stop 
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            print(f" ✔ val_loss melhorou → {best_val_loss:.6f}")
        else:
            patience_count += 1
            print(f" ✘ sem melhora há {patience_count}/{patience} épocas")
            if patience_count >= patience:
                print(f"\nEarly stopping acionado na época {epoch}.")
                break

    # Recuperar os pesos salvos
    model.load_state_dict(best_weights)
    print("Melhores pesos restaurados.")

    # Apresenta a curva para treinamento
    epochs_ran = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_ran, train_losses, 'r', label="Train Loss")
    plt.plot(epochs_ran, val_losses, 'b', label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curvas de Loss — LeNet-5 MNIST")
    plt.legend()
    plt.grid()
    plt.xlim([1, len(train_losses)])
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    #plt.show()

    # Faz o teste do modelo
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y)

    ypred  = torch.cat(all_preds).numpy()
    ytrue  = torch.cat(all_labels).numpy()

    mtrz_cfs = confusion_matrix(ytrue, ypred)
    test_acc  = accuracy_score(ytrue, ypred)
    print(f"\nTeste final | acc: {test_acc:.4f}")
    print(mtrz_cfs) 
