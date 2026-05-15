import matplotlib.pyplot as plt
import numpy as np
from bfs import bfs
from dfs import dfs
from grafos import lab2grafo, imprimir_grafo

labirinto = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
]

grafo = lab2grafo(labirinto)
imprimir_grafo(grafo)
caminho = dfs(grafo, (0,0), (4,4))

lab = np.array(labirinto)
plt.figure(figsize=(5, 5))
plt.imshow(lab, cmap='gray_r')
plt.xticks(np.arange(-0.5, 5, 1), [])
plt.yticks(np.arange(-0.5, 5, 1), [])
plt.grid(color='gray', linewidth=0.5)
plt.scatter(0, 0, c='green', s=200, zorder=5, label='Início')
plt.scatter(4, 4, c='red',   s=200, zorder=5, label='Fim')
for (r, c) in caminho[1:-1]:
    plt.scatter(c, r, c='orange', s=200, zorder=5)
plt.legend()
plt.tight_layout()
plt.show()