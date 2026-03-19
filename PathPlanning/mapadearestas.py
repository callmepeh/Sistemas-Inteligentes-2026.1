import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import Polygon, LineString
import random


# defininindo as contantes do mapa
LARGURA_MAPA = 1000
ALTURA_MAPA = 500

INICIO = (20, 20)
FIM = (500, 500)

NUM_OBSTACULOS = 100
TAMANHO_TRIANGULO = 12


# Criando os triangulos (obstáculos) a partir do centro e do lado
def criar_triangulo(centro, lado):
    cx, cy = centro
    h = (np.sqrt(3) / 2) * lado

    return [
        (cx, cy + (2/3)*h),
        (cx - lado/2, cy - (1/3)*h),
        (cx + lado/2, cy - (1/3)*h)
    ]


#Gerar obstáculos aleatórios, evitando colisões entre eles
def gerar_obstaculos():
    obstaculos = []
    poligonos = []

    tentativas = 0
    max_tentativas = NUM_OBSTACULOS * 200

    while len(obstaculos) < NUM_OBSTACULOS and tentativas < max_tentativas:
        tentativas += 1

        cx = random.uniform(TAMANHO_TRIANGULO, LARGURA_MAPA - TAMANHO_TRIANGULO)
        cy = random.uniform(TAMANHO_TRIANGULO, ALTURA_MAPA - TAMANHO_TRIANGULO)

        triangulo = criar_triangulo((cx, cy), TAMANHO_TRIANGULO)
        poligono = Polygon(triangulo)

        # evita colisão
        if not any(poligono.intersects(p) for p in poligonos):
            obstaculos.append(triangulo)
            poligonos.append(poligono)

    return obstaculos, poligonos


# GRAFO DE VISIBILIDADE (Juliana estaria orgulhosa)
def construir_grafo(obstaculos, poligonos):
    vertices = [INICIO, FIM]

    for tri in obstaculos:
        vertices.extend(tri)

    arestas = []

    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            p1, p2 = vertices[i], vertices[j]
            linha = LineString([p1, p2])

            visivel = True
            for pol in poligonos:
                if linha.crosses(pol):
                    visivel = False
                    break

            if visivel:
                arestas.append((p1, p2))

    return vertices, arestas


# PLOTAGEM
def plotar_mapa(obstaculos, vertices, arestas):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f5f5f5")

    # obstáculos
    for tri in obstaculos:
        patch = patches.Polygon(
            tri,
            closed=True,
            edgecolor="black",
            facecolor="#4a90e2",
            alpha=0.7
        )
        ax.add_patch(patch)

    # arestas (linhas do grafo)
    for (p1, p2) in arestas:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5, alpha=0.3)

    # vértices
    xs, ys = zip(*vertices)
    ax.scatter(xs, ys, s=10)

    # início e fim
    ax.scatter(*INICIO, s=120, label="Início", zorder=3)
    ax.scatter(*FIM, s=120, label="Fim", zorder=3)

    # limites fixos (mais estável)
    ax.set_xlim(0, LARGURA_MAPA)
    ax.set_ylim(0, ALTURA_MAPA)

    ax.set_aspect('equal')
    ax.set_title("Mapa com Obstáculos e Grafo de Visibilidade")

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    plt.show()


# MAINzinha
if __name__ == "__main__":
    obstaculos, poligonos = gerar_obstaculos()
    vertices, arestas = construir_grafo(obstaculos, poligonos)
    plotar_mapa(obstaculos, vertices, arestas)