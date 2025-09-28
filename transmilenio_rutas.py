"""
transmilenio_rutas.py
Sistema basado en conocimiento (KB de hechos lógicos) para buscar la mejor
ruta entre estaciones del TransMilenio (ejemplo: Portal Eldorado / Aeropuerto).
Soporta: parseo de KB (edge, coord, oneway), forward-chaining, BFS, Dijkstra, A*,
y visualización simple con matplotlib.
"""

import re, math, heapq
from collections import defaultdict, deque
import matplotlib.pyplot as plt

# -----------------------
# PARSER KB (hechos estilo Prolog)
# -----------------------
def parse_fact(line):
    line = line.strip()
    if not line or line.startswith('%') or line.startswith('#'):
        return None
    if line.endswith('.'):
        line = line[:-1]
    m = re.match(r"(\w+)\s*\(\s*(.*)\s*\)\s*$", line)
    if not m:
        raise ValueError(f"Can't parse fact: {line}")
    pred = m.group(1)
    args = m.group(2)
    parts = []
    cur = ''
    inq = False
    qchar = ''
    for ch in args:
        if ch in ("'", '"'):
            if not inq:
                inq = True; qchar = ch; cur += ch
            elif qchar == ch:
                inq = False; cur += ch
            else:
                cur += ch
        elif ch == ',' and not inq:
            parts.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    cleaned = []
    for p in parts:
        p = p.strip()
        if (p.startswith("'") and p.endswith("'")) or (p.startswith('"') and p.endswith('"')):
            cleaned.append(p[1:-1])
        else:
            try:
                if '.' in p:
                    cleaned.append(float(p))
                else:
                    cleaned.append(int(p))
            except:
                cleaned.append(p)
    return pred, cleaned

def parse_kb_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    facts = defaultdict(list)
    for ln in lines:
        parsed = parse_fact(ln)
        if parsed:
            pred, args = parsed
            facts[pred].append(args)
    return facts

# -----------------------
# Construcción de grafo desde facts
# -----------------------
def build_graph_from_kb(facts):
    graph = defaultdict(list)
    coords = {}
    for args in facts.get('coord', []):
        name, x, y = args[0], float(args[1]), float(args[2])
        coords[name] = (x, y)
    for args in facts.get('edge', []):
        if len(args) < 5:
            raise ValueError("edge debe tener 5 argumentos: edge(A,B,modo,costo,tiempo)")
        a, b, modo, costo, tiempo = args[0], args[1], args[2], float(args[3]), float(args[4])
        attrs = {'mode': modo, 'cost': costo, 'time': tiempo}
        graph[a].append((b, attrs))
    oneway = set(tuple(x[:2]) for x in facts.get('oneway', []))
    # añadir reversas si no existen y no están en oneway
    for a in list(graph.keys()):
        for b, attrs in list(graph[a]):
            rev_exists = any(nb == a for nb, _ in graph.get(b, []))
            if not rev_exists and (a, b) not in oneway:
                graph[b].append((a, attrs.copy()))
    return graph, coords

# -----------------------
# Forward chaining: reachable por BFS desde cada nodo (cierre transitivo)
# -----------------------
def forward_reachable(facts):
    graph, _ = build_graph_from_kb(facts)
    reachable = set()
    paths = {}
    for s in graph.keys():
        q = deque([(s, [s])])
        seen = {s}
        while q:
            node, path = q.popleft()
            for nb, _ in graph.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    newpath = path + [nb]
                    reachable.add((s, nb))
                    paths[(s, nb)] = newpath
                    q.append((nb, newpath))
    return reachable, paths

# -----------------------
# Algoritmos de búsqueda
# -----------------------
def bfs_hops(graph, start, goal):
    q = deque([(start, [start])])
    seen = {start}
    while q:
        node, path = q.popleft()
        if node == goal:
            return path
        for nb, _ in graph.get(node, []):
            if nb not in seen:
                seen.add(nb)
                q.append((nb, path + [nb]))
    return None

def dijkstra(graph, start, goal, weight='time'):
    dist = {start: 0}
    prev = {}
    heap = [(0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float('inf')): continue
        if node == goal:
            path = []
            cur = goal
            while cur != start:
                path.append(cur); cur = prev[cur]
            path.append(start); path.reverse()
            return path, d
        for nb, attrs in graph.get(node, []):
            w = attrs.get(weight, 1)
            nd = d + w
            if nd < dist.get(nb, float('inf')):
                dist[nb] = nd; prev[nb] = node
                heapq.heappush(heap, (nd, nb))
    return None, float('inf')

def heuristic(a, b, coords):
    ax, ay = coords.get(a, (0,0)); bx, by = coords.get(b, (0,0))
    return math.hypot(ax-bx, ay-by)

def astar(graph, start, goal, coords=None, weight='time'):
    if coords is None: coords = {}
    openh = []
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal, coords)}
    heapq.heappush(openh, (fscore[start], start))
    came = {}
    while openh:
        _, curr = heapq.heappop(openh)
        if curr == goal:
            path = []
            node = goal
            while node in came:
                path.append(node); node = came[node]
            path.append(start); path.reverse()
            return path, gscore[goal]
        for nb, attrs in graph.get(curr, []):
            tentative = gscore[curr] + attrs.get(weight, 1)
            if tentative < gscore.get(nb, float('inf')):
                came[nb] = curr
                gscore[nb] = tentative
                f = tentative + heuristic(nb, goal, coords)
                heapq.heappush(openh, (f, nb))
    return None, float('inf')

# -----------------------
# Util: describir ruta
# -----------------------
def describe_route(path, graph):
    if not path: return "No hay ruta."
    lines = []
    tc = 0; tt = 0
    for i in range(len(path)-1):
        a, b = path[i], path[i+1]
        attrs = None
        for nb, at in graph[a]:
            if nb == b: attrs = at; break
        mode = attrs.get('mode') if attrs else '?'
        cost = attrs.get('cost',0) if attrs else 0
        time = attrs.get('time',0) if attrs else 0
        tc += cost; tt += time
        lines.append(f"{a} -> {b} via {mode} (cost={cost}, time={time})")
    lines.append(f"TOTAL cost={tc}, time={tt}")
    return "\n".join(lines)

# -----------------------
# Visualización simple con matplotlib
# -----------------------
def plot_graph_and_route(graph, coords, route=None, title="Mapa TransMilenio (ejemplo)"):
    plt.figure(figsize=(8,6))
    # dibuja nodos
    for n, (x,y) in coords.items():
        plt.scatter(x, y)
        plt.text(x+0.02, y+0.02, n, fontsize=9)
    # dibuja aristas
    for a, nbrs in graph.items():
        xa, ya = coords.get(a, (None, None))
        for b, _ in nbrs:
            xb, yb = coords.get(b, (None, None))
            if xa is not None and xb is not None:
                plt.plot([xa, xb], [ya, yb], linewidth=0.8)
    # resalta ruta
    if route:
        for i in range(len(route)-1):
            a, b = route[i], route[i+1]
            xa, ya = coords.get(a); xb, yb = coords.get(b)
            plt.plot([xa, xb], [ya, yb], linewidth=3)
    plt.title(title)
    plt.axis('off')
    plt.show()

# -----------------------
# Demo: carga KB y ejecuta búsquedas
# -----------------------
def demo_from_kb(path_kb, start, goal, visualize=True):
    facts = parse_kb_file(path_kb)
    graph, coords = build_graph_from_kb(facts)
    print("Nodos (muestra):", sorted(graph.keys()))
    reachable, paths = forward_reachable(facts)
    print(f"reachable examples (hasta 10): {list(reachable)[:10]}")
    path_bfs = bfs_hops(graph, start, goal)
    print("BFS (mínimos saltos):", path_bfs)
    p_dij, t_dij = dijkstra(graph, start, goal, weight='time')
    print("Dijkstra (min tiempo):", p_dij, "tiempo=", t_dij)
    p_ast, t_ast = astar(graph, start, goal, coords=coords, weight='time')
    print("A* (heurística coords):", p_ast, "costo_est=", t_ast)
    print("\nDetalle Dijkstra:\n", describe_route(p_dij, graph))
    if visualize:
        plot_graph_and_route(graph, coords, route=p_dij, title=f"Ruta {start} -> {goal} (Dijkstra)")
    return {'graph': graph, 'coords': coords, 'paths': {'bfs': path_bfs, 'dijkstra': p_dij, 'astar': p_ast}}

# -----------------------
# Si se ejecuta directamente: demo usando kb_bogota.txt
# -----------------------
if __name__ == "__main__":
    # cambio los nombres start/goal a conveniencia
    demo_from_kb('kb_bogota.txt', start='Portal_Eldorado', goal='Aeropuerto_El_Dorado')
