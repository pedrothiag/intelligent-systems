from dfs import dfs
from grafos import imprimir_grafo

# Exemplo de Grafo
grafo_exemplo = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['E', 'F'],
    'D': ['G', 'H'],
    'E': ['I', 'J'],
    'F': ['K', 'L']
}

# Executando o algoritmo
imprimir_grafo(grafo_exemplo)
caminho = dfs(grafo_exemplo, 'A', objetivo='J')
print(f"Nós visitados: {caminho}")