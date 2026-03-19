import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import Polygon, LineString
import random


# CONFIGURAÇÕES
MAP_SIZE = (100, 100)
NUM_OBSTACLES = 500

START = (0, 0)
GOAL = (1000, 1000)

TRIANGLE_SIDE = 10


# TRIÂNGULO
def create_triangle(center, side):
    cx, cy = center
    h = (np.sqrt(3) / 2) * side

    return [
        (cx, cy + (2/3)*h),
        (cx - side/2, cy - (1/3)*h),
        (cx + side/2, cy - (1/3)*h)
    ]


# MAPA
def generate_map(num_obstacles, map_size, start, goal):
    obstacles = []
    polygons = []

    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    length = np.hypot(dx, dy)

    ux, uy = dx/length, dy/length
    perp_x, perp_y = -uy, ux

    corridor_width = 20

    attempts = 0
    max_attempts = num_obstacles * 300

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1

        t = random.uniform(0, 1)
        base_x = start[0] + t * dx
        base_y = start[1] + t * dy

        offset = random.uniform(-corridor_width, corridor_width)
        cx = base_x + offset * perp_x
        cy = base_y + offset * perp_y

        if not (TRIANGLE_SIDE < cx < map_size[0] - TRIANGLE_SIDE and
                TRIANGLE_SIDE < cy < map_size[1] - TRIANGLE_SIDE):
            continue

        triangle = create_triangle((cx, cy), TRIANGLE_SIDE)
        poly = Polygon(triangle)

        if not any(poly.intersects(p) for p in polygons):
            obstacles.append(triangle)
            polygons.append(poly)

    return obstacles, polygons


# GRAFO DE VISIBILIDADE
def build_visibility_graph(obstacles, polygons):
    vertices = [START, GOAL]

    # adiciona vértices dos triângulos
    for tri in obstacles:
        vertices.extend(tri)

    edges = []

    # testa todos os pares
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            p1 = vertices[i]
            p2 = vertices[j]

            line = LineString([p1, p2])

            visible = True

            for poly in polygons:
                # se a linha cruza o interior → bloqueado
                if line.crosses(poly) or line.within(poly):
                    visible = False
                    break

            if visible:
                edges.append((p1, p2))

    return vertices, edges


# PLOT
def plot_map(obstacles, polygons, vertices, edges):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f7f7f7")

    # obstáculos
    for tri in obstacles:
        patch = patches.Polygon(
            tri, closed=True,
            edgecolor="black",
            facecolor="#4a90e2",
            alpha=0.7
        )
        ax.add_patch(patch)

    # arestas (grafo)
    for (p1, p2) in edges:
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        ax.plot(x, y, linewidth=0.5, alpha=0.4)

    # vértices
    xs, ys = zip(*vertices)
    ax.scatter(xs, ys, s=20)

    # início e fim
    ax.scatter(*START, s=120, label="Início", zorder=3)
    ax.scatter(*GOAL, s=120, label="Fim", zorder=3)

    ax.set_xlim(0, MAP_SIZE[0])
    ax.set_ylim(0, MAP_SIZE[1])
    ax.set_aspect('equal')

    ax.set_title("Grafo de Visibilidade")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()


# MAINzinha
if __name__ == "__main__":
    obstacles, polygons = generate_map(NUM_OBSTACLES, MAP_SIZE, START, GOAL)
    vertices, edges = build_visibility_graph(obstacles, polygons)
    plot_map(obstacles, polygons, vertices, edges)