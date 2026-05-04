from collections import deque
from grafos import sucessores, imprimir_grafo

# Algoritmo de Busca em Profundidade (DFS)
def dfs(grafo, node_inicial, objetivo=None):
    visitados = []                                              # Lista para armazenar os vértices visitados, iniciando vazia
    pilha = [node_inicial]                                      # Pilha para armazenar os vértices a serem visitados, iniciando com o nó inicial
    
    while pilha:                                                # Enquanto houver vértices na pilha, continue a busca
        vertice = pilha.pop()                                   # Remove o vértice do topo da pilha

        if vertice not in visitados:                            # Verifica se o vértice já foi visitado
            visitados.append(vertice)                           # Marca o vértice como visitado
            print(f"Visitando nó: {vertice} \t Pilha: {list(pilha)}")
        
            if vertice == objetivo:                             # Verifica se o vértice é o objetivo
                print(f"Objetivo {objetivo} encontrado!")
                return list(visitados)                          # Retorna a lista de vértices visitados (incluindo o objetivo)
        
            estados_proximos = sucessores(grafo, vertice)       # Obtém os estados próximos (vizinhos) do vértice atual

            if estados_proximos:                                # Se houver estados próximos, adicione-os à pilha para serem visitados posteriormente
                for vizinho in reversed(estados_proximos):      # Para cada vizinho, verifique se ele já foi visitado antes de adicioná-lo à pilha
                    if vizinho not in visitados:                # Se o vizinho ainda não foi visitado, adicione-o à pilha
                        pilha.append(vizinho)                   # Apenda o vizinho à pilha
                        
    return list(visitados)                                      # Retorna a lista de vértices visitados (sem encontrar o objetivo)

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
print(f"Vértices visitados: {caminho}")