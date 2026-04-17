from collections import deque
from grafos import imprimir_grafo, sucessores

# Algoritmo de Busca em Largura (BFS)
def bfsback(grafo, node_inicial, objetivo=None):
    visitados = []                                                  # Lista para armazenar os vértices visitados, iniciando vazia
    fila = deque([node_inicial])                                    # Fila para armazenar os vértices a serem visitados, iniciando com o nó inicial
    predecessores = {node_inicial: None}                            # Dicionário para armazenar os predecessores de cada nó, iniciando com o nó inicial sem predecessor
    caminho = []                                                    # Lista para armazenar o caminho do nó inicial ao objetivo
    
    while fila:                                                     # Enquanto houver vértices na fila, continue a busca
        vertice = fila.popleft()                                    # Remove o vértice do início da fila

        if vertice not in visitados:                                # Verifica se o vértice já foi visitado (É necessário para evitar ciclos e visitas repetidas)
            visitados.append(vertice)                               # Marca o vértice como visitado
            print(f"Visitando nó: {vertice}")
        
            if vertice == objetivo:                                 # Verifica se o vértice é o objetivo
                print(f"Objetivo {objetivo} encontrado!")
                atual = objetivo                                    # Variável para reconstruir o caminho do nó inicial ao objetivo
                while atual is not None:                            # Enquanto houver um predecessor, adicione o nó atual ao caminho e mova para o predecessor
                    caminho.append(atual)                           # Adiciona o nó atual ao caminho
                    atual = predecessores[atual]                    # Move para o predecessor do nó atual
                caminho.reverse()                                   # Inverte o caminho para obter a ordem correta do nó inicial ao objetivo
                return caminho, list(visitados)                     # Retorna o caminho encontrado e a lista de vértices visitados
        
            estados_proximos = sucessores(grafo, vertice)           # Obtém os estados próximos (vizinhos) do vértice atual

            if estados_proximos:                                    # Adicione-os à fila para serem visitados posteriormente
                for vizinho in estados_proximos:                    # Para cada vizinho, verifique se ele já foi visitado antes de adicioná-lo à fila
                    if vizinho not in visitados:                    # Se o vizinho ainda não foi visitado, adicione-o à fila e registre seu predecessor
                        fila.append(vizinho)                        # Apenda o vizinho à fila
                        if vizinho not in predecessores:            # Registra o predecessor do vizinho apenas se ele ainda não tiver um predecessor registrado
                            predecessores[vizinho] = vertice        # Registra o predecessor do vizinho como o vértice atual
    
    caminho.reverse()
    return caminho, list(visitados)

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
caminho, visitados = bfsback(grafo_exemplo, 'A', objetivo='J')
print(f"Caminho encontrado: {caminho}")
print(f"Vértices visitados: {visitados}")