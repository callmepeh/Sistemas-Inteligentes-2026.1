#tem que baixar o shapely: pip install shapely
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import Polygon
import random



MAP_SIZE = (100, 100)
NUM_OBSTACLES = 20

START = (10, 10)
GOAL = (90, 90)

TRIANGLE_SIDE = 10  # 🔥 tamanho FIXO

#funcoes para criar os triangulos
def create_triangle(center, side):
    cx, cy = center
    h = (np.sqrt(3) / 2) * side

    return [
        (cx, cy + (2/3)*h),              # topo
        (cx - side/2, cy - (1/3)*h),     # esquerda
        (cx + side/2, cy - (1/3)*h)      # direita
    ]


#runcao para gerar o mapa com obstáculos ao longo do caminho entre início e fim
def generate_map(num_obstacles, map_size, start, goal):
    obstacles = []
    polygons = []

    attempts = 0
    max_attempts = num_obstacles * 300

    # vetor direção
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    length = np.hypot(dx, dy)

    # vetor unitário
    ux = dx / length
    uy = dy / length

    # vetor perpendicular (largura da faixa)
    perp_x = -uy
    perp_y = ux

    corridor_width = 20

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1

        side = TRIANGLE_SIDE
        h = (np.sqrt(3) / 2) * side

        # ponto ao longo da linha
        t = random.uniform(0, 1)
        base_x = start[0] + t * dx
        base_y = start[1] + t * dy

        # deslocamento lateral
        offset = random.uniform(-corridor_width, corridor_width)
        cx = base_x + offset * perp_x
        cy = base_y + offset * perp_y

        # mantém dentro do mapa
        if not (side < cx < map_size[0] - side and side < cy < map_size[1] - side):
            continue

        triangle = create_triangle((cx, cy), side)
        poly = Polygon(triangle)

        # verifica colisão
        if not any(poly.intersects(p) for p in polygons):
            obstacles.append(triangle)
            polygons.append(poly)

    return obstacles

# PLOTAGEM
def plot_map(obstacles, map_size):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_facecolor("#f7f7f7")

    # obstáculos
    for tri in obstacles:
        patch = patches.Polygon(
            tri,
            closed=True,
            linewidth=1.5,
            edgecolor="black",
            facecolor="#4a90e2",
            alpha=0.7
        )
        ax.add_patch(patch)

    # início e fim
    ax.scatter(*START, s=120, zorder=3, label="Início")
    ax.scatter(*GOAL, s=120, zorder=3, label="Fim")

    # limites
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.set_aspect('equal')

    # estilo
    ax.set_title("Mapa com Obstáculos no Caminho")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    # remove bordas
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()

# MAINzinha
if __name__ == "__main__":
    obstacles = generate_map(NUM_OBSTACLES, MAP_SIZE, START, GOAL)
    plot_map(obstacles, MAP_SIZE)