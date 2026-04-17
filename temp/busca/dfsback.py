from collections import deque
from grafos import sucessores, imprimir_grafo

# Algoritmo de Busca em Profundidade (DFS)
def dfsback(grafo, node_inicial, objetivo=None):
    visitados = []                                                  # Lista para armazenar os vértices visitados, iniciando vazia
    pilha = [node_inicial]                                          # Pilha para armazenar os vértices a serem visitados, iniciando com o nó inicial
    predecessores = {node_inicial: None}                            # Dicionário para armazenar os predecessores de cada nó, iniciando com o nó inicial sem predecessor
    caminho = []                                                    # Lista para armazenar o caminho do nó inicial ao objetivo
    
    while pilha:                                                    # Enquanto houver vértices na pilha, continue a busca
        vertice = pilha.pop()                                       # Remove o vértice do topo da pilha

        if vertice not in visitados:                                # Verifica se o vértice já foi visitado
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

            if estados_proximos:                                    # Adicione-os à pilha para serem visitados posteriormente
                for vizinho in reversed(estados_proximos):          # Para cada vizinho, verifique se ele já foi visitado antes de adicioná-lo à pilha
                    if vizinho not in visitados:                    # Se o vizinho ainda não foi visitado, adicione-o à pilha e registre seu predecessor
                        pilha.append(vizinho)                       # Apenda o vizinho à pilha
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
caminho, visitados = dfsback(grafo_exemplo, 'A', objetivo='J')
print(f"Caminho encontrado: {caminho}")
print(f"Vértices visitados: {visitados}")