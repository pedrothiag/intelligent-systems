from collections import deque
from grafos import imprimir_grafo, sucessores

# Algoritmo de Busca em Largura (BFS)
def bfs(grafo, node_inicial, objetivo=None):
    visitados = []                                                  # Lista para armazenar os vértices visitados, iniciando vazia
    fila = deque([node_inicial])                                    # Fila para armazenar os vértices a serem visitados, iniciando com o nó inicial
    
    while fila:                                                     # Enquanto houver vértices na fila, continue a busca
        vertice = fila.popleft()                                    # Remove o vértice do início da fila

        if vertice not in visitados:                                # Verifica se o vértice já foi visitado (É necessário para evitar ciclos e visitas repetidas)
            visitados.append(vertice)                               # Marca o vértice como visitado
            print(f"Visitando nó: {vertice} \t Fila: {list(fila)}")
        
            if vertice == objetivo:                                 # Verifica se o vértice é o objetivo
                print(f"Objetivo {objetivo} encontrado!")
                return list(visitados)                              # Retorna a lista de vértices visitados (incluindo o objetivo)
        
            estados_proximos = sucessores(grafo, vertice)           # Obtém os estados próximos (vizinhos) do vértice atual

            if estados_proximos:                                    # Se houver estados próximos, adicione-os à fila para serem visitados posteriormente
                for vizinho in estados_proximos:                    # Para cada vizinho, verifique se ele já foi visitado antes de adicioná-lo à fila
                    if vizinho not in visitados:                    # Se o vizinho ainda não foi visitado, adicione-o à fila para ser visitado posteriormente
                        fila.append(vizinho)                        # Apenda o vizinho à fila para ser visitado posteriormente
    return list(visitados)                                          # Retorna a lista de vértices visitados (sem encontrar o objetivo)

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
caminho = bfs(grafo_exemplo, 'A', objetivo='J')
print(f"Vértices visitados: {caminho}")