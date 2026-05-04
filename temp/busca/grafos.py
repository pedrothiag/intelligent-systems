# Função para obter os sucessores (vizinhos) de um vértice no grafo
def sucessores(grafo, no):
    if no not in grafo:
        return None
    return grafo[no]

# Função para imprimir o grafo
def imprimir_grafo(grafo):
    for no, vizinhos in grafo.items():
        print(f"{no} → {vizinhos}")

# Função para converter um labirinto (matriz) em um grafo
def lab2grafo(lab):
    linhas, colunas = len(lab), len(lab[0])
    grafo = {}
    for i in range(linhas):
        for j in range(colunas):
            if lab[i][j] == 0:  # Verifica se é um caminho
                vizinhos = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni = i + di 
                    nj = j + dj
                    if (0 <= ni < linhas) and (0 <= nj < colunas) and (lab[ni][nj]) == 0:
                        vizinhos.append((ni, nj))
                grafo[(i, j)] = vizinhos
    return grafo