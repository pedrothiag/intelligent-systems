from collections import deque
from grafos import sucessores

# Algoritmo de Busca em Largura (BFS)
def bfs(grafo, node_inicial, objetivo=None):
    visitados = []
    fila = deque([node_inicial])
    
    # Enquanto houver vértices na fila, continue a busca
    while fila:
        # Remove o vértice da frente da fila
        vertice = fila.popleft()

        # Verifica se o vértice já foi visitado (É necessário para evitar ciclos e visitas repetidas)
        if vertice not in visitados:
            # Marca o vértice como visitado
            visitados.append(vertice)
            print(f"Visitando nó: {vertice} \t Fila: {list(fila)}")
        
            # Verifica se o vértice é o objetivo
            if vertice == objetivo:
                print(f"Objetivo {objetivo} encontrado!")
                return list(visitados)
        
            # Obtém os estados próximos (vizinhos) do vértice atual
            estados_proximos = sucessores(grafo, vertice)

            # Se houver estados próximos, adicione-os à fila para serem visitados posteriormente
            if estados_proximos:
                fila.extend(estados_proximos)
    return list(visitados)