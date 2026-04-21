import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Função de ativação
def sigmoid(z):
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

# Derivada de função de ativação
def sigmoid_deriv(a):
    return a * (1 - a)

# Classe para a rede MLP
class MLP:
    def __init__(self, n_input, n_hidden, n_output=1, lr=0.1):
        self.n_input  = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr       = lr
        self._init_weights()

    # Inicialização de pesos
    def _init_weights(self):
        lim1 = np.sqrt(6 / (self.n_input  + self.n_hidden))
        lim2 = np.sqrt(6 / (self.n_hidden + self.n_output))
        self.W1 = np.random.uniform(-lim1, lim1, (self.n_hidden, self.n_input))
        self.b1 = np.zeros(self.n_hidden)
        self.W2 = np.random.uniform(-lim2, lim2, (self.n_output, self.n_hidden))
        self.b2 = np.zeros(self.n_output)

    # Fase forward
    def forward(self, x):
        self._x  = x
        self._u1 = self.W1 @ x + self.b1
        self._a1 = sigmoid(self._u1)
        self._u2 = self.W2 @ self._a1 + self.b2
        self._a2 = sigmoid(self._u2)
        return self._a2

    # Fase backward
    def backward(self, y):
        delta2 = (self._a2 - y) * sigmoid_deriv(self._a2)
        delta1 = (self.W2.T @ delta2) * sigmoid_deriv(self._a1)
        self.W2 -= self.lr * np.outer(delta2, self._a1)
        self.b2 -= self.lr * delta2
        self.W1 -= self.lr * np.outer(delta1, self._x)
        self.b1 -= self.lr * delta1
        return float(0.5 * np.sum((self._a2 - y) ** 2))

    # Função de treinamento
    def train(self, X, Y, epochs=500):
        n = X.shape[0]
        train_loss = []
        for ep in range(1, epochs + 1):
            idx = np.random.permutation(n)
            total = 0.0
            for i in idx:
                self.forward(X[i])
                total += self.backward(Y[i])
            train_loss.append(total / n)
        return train_loss

    # Função para predição
    def predict(self, X):
        return np.array([np.argmax(self.forward(x)) for x in X])

    # Função para calcular a acurácia
    def accuracy(self, X, Y):
        preds = self.predict(X)
        true  = np.argmax(Y, axis=1)
        return np.mean(preds == true)

# Carregamento do dataset    
iris = load_iris()
X = iris.data
y = iris.target
Y = np.eye(3)[y]

# Divisão entre conjunto de treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, stratify=y
)

# Normalização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cria um modelo com 8 neurônios na camada oculta
model = MLP(n_input=4, n_hidden=8, n_output=3, lr=0.1)

# Aplica a função de treinamento
train_loss = model.train(
    X_train, Y_train, epochs=500
)

# Apresenta a curva de treinamento
epoch_vector = np.arange(1, len(train_loss)+1)
plt.figure()
plt.semilogy(epoch_vector, train_loss, 'r', label = "Treinamento")
plt.xlim([1, len(train_loss)])
plt.xlabel("Época de treinamento")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()

# Calcula a acurácia do modelo
y_pred = model.predict(X_test)
y_true = np.argmax(Y_test,  axis=1)
print(f"Acurácia = {accuracy_score(y_true, y_pred)}")
print("Matriz de Confusão:")
print(confusion_matrix(y_true, y_pred))