from collections import deque
from grafos import sucessores, imprimir_grafo

# Algoritmo de Busca em Profundidade (DFS)
def dfs(grafo, node_inicial, objetivo=None):
    #visitados = set()
    visitados = []
    pilha = [node_inicial]
    
    # Enquanto houver vértices na pilha, continue a busca
    while pilha:
        # Remove o vértice do topo da pilha
        vertice = pilha.pop()

        # Verifica se o vértice já foi visitado (É necessário para evitar ciclos e visitas repetidas)
        if vertice not in visitados:
            # Marca o vértice como visitado
            visitados.append(vertice)
            print(f"Visitando nó: {vertice} \t Pilha: {list(pilha)}")
        
            # Verifica se o vértice é o objetivo
            if vertice == objetivo:
                print(f"Objetivo {objetivo} encontrado!")
                return list(visitados)
        
            # Obtém os estados próximos (vizinhos) do vértice atual
            estados_proximos = sucessores(grafo, vertice)

            # Se houver estados próximos, adicione-os à pilha para serem visitados posteriormente
            if estados_proximos:
                #pilha.extend(estados_proximos)
                pilha.extend(reversed(estados_proximos))        # Garantir que a ordem sempre seja feita pela esquerda
    return list(visitados)

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