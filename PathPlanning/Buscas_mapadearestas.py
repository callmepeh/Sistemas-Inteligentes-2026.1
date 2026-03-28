import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from shapely.geometry import Polygon, LineString
import random


# CONFIGURAÇÕES
MAP_SIZE = (100, 100)
NUM_OBSTACLES = 20

START = (5, 5)
GOAL = (95, 95)

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

    corridor_width = 30

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

def safe_paths(obstacles, polygons, start, goal):
   paths = {"Vértice":"y" "x"}
   for tri in obstacles:
        for vert in tri:
            if vert == start or vert == goal:
                continue
            line_to_start = LineString([vert, start])
            line_to_goal = LineString([vert, goal])
            if not any(line_to_start.crosses(p) or line_to_start.within(p) for p in polygons):
                paths[vert] = "start"
            if not any(line_to_goal.crosses(p) or line_to_goal.within(p) for p in polygons):
                paths[vert] = "goal"

# def BFT(graph, start, goal, looking_for):
#     visited = set()
#     queue = [start]

#     while queue:
#         current = queue.pop(0)
#         if current == goal:
#             return True
#         visited.add(current)

#         for neighbor in graph[current]:
#             if neighbor not in visited and neighbor not in queue:
#                 queue.append(neighbor)

#     return False

# def DFS(graph, start, goal, looking_for):
#     visited = set()
#     stack = [start]

#     while stack:
#         current = stack.pop()
#         if current == goal:
#             return True
#         visited.add(current)

#         for neighbor in graph[current]:
#             if neighbor not in visited and neighbor not in stack:
#                 stack.append(neighbor)

#     return False

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

def build_search_graph(vertices, edges):
    graph = {v: [] for v in vertices}
    for (u, v) in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph

def BFT(graph, start, goal):
    visited = set()
    queue = [start]

    while queue:
        current = queue.pop(0)
        if current == goal:
            return True
        visited.add(current)

        for neighbor in graph[current]:
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)

    return False

def build_bft_path(graph, start, goal):
    visited = set()
    queue = [start]
    parent = {start: None}
    visited_order = []  # lista para ordem de visita

    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            visited_order.append(current)  # adiciona à ordem de visita
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1], visited_order  # retorna caminho e ordem de visita
        else:
            continue

        for neighbor in graph[current]:
            if neighbor not in visited and neighbor not in queue:
                parent[neighbor] = current
                queue.append(neighbor)

    return None, visited_order  # sem caminho, mas retorna ordem de visita

def BFS(graph, start, goal):
    visited = set()
    queue = [start]
    parent = {start: None}

    while queue:
        current = queue.pop(0)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]  # caminho encontrado
        visited.add(current)

        for neighbor in graph[current]:
            if neighbor not in visited and neighbor not in queue:
                parent[neighbor] = current
                queue.append(neighbor)

    return None  # sem caminho encontrado

def build_dfs_path(graph, start, goal):
    visited = set()
    stack = [start]
    parent = {start: None}

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]  # caminho encontrado
        else:
            continue

        for neighbor in graph[current]:
            if neighbor not in visited and neighbor not in stack:
                parent[neighbor] = current
                stack.append(neighbor)

    return None  # sem caminho encontrado

# PLOT
def plot_map(obstacles, polygons, vertices, edges):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f7f7f7ff")

    # obstáculos
    for tri in obstacles:
        patch = patches.Polygon(
            tri, closed=True,
            edgecolor="black",
            facecolor="#e2a04a",
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

def plot_search_path(obstacles, polygons, vertices, edges, path, tip_caminho):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f7f7f7ff")

    # obstáculos
    for tri in obstacles:
        patch = patches.Polygon(
            tri, closed=True,
            edgecolor="black",
            facecolor="#e2a04a",
            alpha=0.7
        )
        ax.add_patch(patch)

    # arestas (grafo)
    for (p1, p2) in edges:
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        ax.plot(x, y, linewidth=0.5, alpha=0.4)

    # caminho encontrado
    if path:
        path_xs, path_ys = zip(*path)
        ax.plot(path_xs, path_ys, color="red", linewidth=2, label="Caminho Encontrado", zorder=4)

    # vértices
    xs, ys = zip(*vertices)
    ax.scatter(xs, ys, s=20)

    # início e fim
    ax.scatter(*START, s=120, label="Início", zorder=3)
    ax.scatter(*GOAL, s=120, label="Fim", zorder=3)

    ax.set_xlim(0, MAP_SIZE[0])
    ax.set_ylim(0, MAP_SIZE[1])
    ax.set_aspect('equal')

    ax.set_title("Grafo de Visibilidade com Busca (%s)" %tip_caminho)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()


# def animate_bft_exploration(obstacles, polygons, vertices, edges, visited_order, path):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_facecolor("#f7f7f7ff")

#     # obstáculos
#     for tri in obstacles:
#         patch = patches.Polygon(
#             tri, closed=True,
#             edgecolor="black",
#             facecolor="#e2a04a",
#             alpha=0.7
#         )
#         ax.add_patch(patch)

#     # arestas (grafo)
#     for (p1, p2) in edges:
#         x = [p1[0], p2[0]]
#         y = [p1[1], p2[1]]
#         ax.plot(x, y, linewidth=0.5, alpha=0.4, color='gray')

#     # vértices
#     xs, ys = zip(*vertices)
#     scatter = ax.scatter(xs, ys, s=20, color='blue')

#     # início e fim
#     ax.scatter(*START, s=120, label="Início", zorder=3, color='green')
#     ax.scatter(*GOAL, s=120, label="Fim", zorder=3, color='red')

#     ax.set_xlim(0, MAP_SIZE[0])
#     ax.set_ylim(0, MAP_SIZE[1])
#     ax.set_aspect('equal')
#     ax.set_title("Exploração BFT Passo a Passo")
#     ax.grid(True, linestyle="--", alpha=0.3)
#     ax.legend()

#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     visited_scatter = ax.scatter([], [], s=50, color='orange', label='Visitados', zorder=2)

#     def update(frame):
#         if frame < len(visited_order):
#             current_visited = visited_order[:frame+1]
#             vx, vy = zip(*current_visited) if current_visited else ([], [])
#             visited_scatter.set_offsets(list(zip(vx, vy)))
#         return visited_scatter,

#     ani = animation.FuncAnimation(fig, update, frames=len(visited_order), interval=500, blit=True, repeat=False)
#     plt.show()


# MAINzinha
if __name__ == "__main__":
    obstacles, polygons = generate_map(NUM_OBSTACLES, MAP_SIZE, START, GOAL)
    vertices, edges = build_visibility_graph(obstacles, polygons)
    plot_map(obstacles, polygons, vertices, edges)
    graph = build_search_graph(vertices, edges)
    path, visited_order = build_bft_path(graph, START, GOAL)
    plot_search_path(obstacles, polygons, vertices, edges, path, "BFT")
    path = build_dfs_path(graph, START, GOAL)
    plot_search_path(obstacles, polygons, vertices, edges, path, "DFS")
    #animate_bft_exploration(obstacles, polygons, vertices, edges, visited_order, path)
    