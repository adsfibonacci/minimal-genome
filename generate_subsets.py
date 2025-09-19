import numpy as np
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import pickle


def k_level_nor(layers, unused, overlap=False):
    layer1, layer2 = layers
    unused = list(unused)
    viable = tuple()
    if layer1 > len(unused) or layer2 > len(unused):
        return viable
    nv1 = set(np.random.choice(unused, size=(layer1), replace=False))
    if overlap:
        nv2 = set(np.random.choice(unused, size=(layer2), replace=False))
        viable = tuple(nv1.union(nv2))
    else:
        if layer1 + layer2 > len(unused):
            return viable
        unused = list(set(unused).difference(nv1))
        nv2 = set(np.random.choice(unused, size=(layer2), replace=False))
        viable = tuple(nv1.union(nv2))
    return viable


def generate_layers(n, max_k, base_prob=0.5, decay=0.5, new_elem_prob=0.1,
                    max_extensions=10, scale_factor=5, bad_spawn_prob=.01):
    """
    Generate layered random subsets of {0,1,...,n}, then add one extra
    random set of size n//4 at its correct level.

    Scales to large n by:
      - Sampling at most `max_extensions` new elements per set
      - Thinning each new layer to about scale_factor * sqrt(n) sets
    """
    ground = list(range(n+1))
    layers = {}
    used = set()

    # --- Layer 1: singletons ---
    layer_prob = base_prob
    L1 = []
    for x in ground:
        if random.random() < layer_prob:
            L1.append((x,))
            used.add(x)
    layers[1] = L1

    # --- Higher layers by growth ---
    for k in range(2, max_k+1):
        layer_prob *= decay  # shrink probability
        prev_layer = layers.get(k-1, [])
        new_layer = []
        prev_elems = {e for subset in prev_layer for e in subset}

        for subset in prev_layer:
            # choose a few candidate new elements instead of all
            candidates = [x for x in ground if x not in subset]
            if not candidates:
                continue
            sample_size = min(max_extensions, len(candidates))
            for x in random.sample(candidates, sample_size):
                x = int(x)
                if x not in prev_elems:
                    # chance to introduce a truly "new" element
                    if random.random() < new_elem_prob * (decay ** (k-1)):
                        new_set = tuple(sorted(set(subset) | {x}))
                        if len(new_set) == k:
                            new_layer.append(new_set)
                            used.update(new_set)
                else:
                    # extend with previously seen element
                    if random.random() < layer_prob:
                        new_set = tuple(sorted(set(subset) | {x}))
                        if len(new_set) == k:
                            new_layer.append(new_set)

        if random.random() < bad_spawn_prob:
            bad_size = [k,k]
            while(sum(bad_size) != k):
                bad_size = [int(x) for x in np.random.randint(1, max_k, size=2)]
            bad_layer = [int(x) for x in k_level_nor(bad_size, set(ground).difference(used), overlap=False)]
            # print(bad_size, sum(bad_size), sorted(np.array(bad_layer)))
            new_layer.append(tuple(bad_layer))
                
        # deduplicate
        new_layer = list(set(new_layer))

        # --- adaptive thinning ---
        max_allowed = int(scale_factor * math.sqrt(n))
        if len(new_layer) > max_allowed:
            keep_prob = max_allowed / len(new_layer)
            new_layer = [s for s in new_layer if random.random() < keep_prob]

        if new_layer:
            layers[k] = new_layer

    # --- Add special random set of size n//4 ---
    seed_size = max(2, n // 4)  # ensure at least size 2
    if seed_size <= max_k:
        seed_set = tuple(sorted(int(x) for x in random.sample(ground, seed_size)))
        layers.setdefault(seed_size, [])
        if seed_set not in layers[seed_size]:
            layers[seed_size].append(seed_set)

    return layers


def visualize_layers(layers):
    """
    Visualize layered subsets as a directed graph.
    Args:
        layers (dict): layer index -> list of sets (as tuples)
    """
    G = nx.DiGraph()

    # --- Add nodes ---
    for k, sets in layers.items():
        for s in sets:
            s = tuple(sorted(s))  # ensure canonical form
            G.add_node(s, layer=k)

    # --- Add edges between consecutive layers ---
    for k in range(1, max(layers.keys())):
        if k not in layers or (k+1) not in layers:
            continue
        for s in layers[k]:
            s = tuple(sorted(s))
            for t in layers[k+1]:
                t = tuple(sorted(t))
                if set(s).issubset(t) and len(t) == len(s) + 1:
                    G.add_edge(s, t)

    # --- Layout: evenly spaced rows ---
    pos = {}
    max_layer = max(layers.keys())
    for k, sets in layers.items():
        unique_sets = [tuple(sorted(s)) for s in sets]
        unique_sets = list(dict.fromkeys(unique_sets))
        n_nodes = len(unique_sets)
        x_spacing = 1.0 / (n_nodes + 1)
        for i, s in enumerate(unique_sets):
            pos[s] = (i * x_spacing + 0.05, -k)  # y = -k so layer 1 is on top

    # --- Draw graph ---
    plt.figure(figsize=(10, 6))
    nx.draw(
        G, pos,
        with_labels=False,
        node_size=1200,
        node_color="lightblue",
        font_size=8,
        arrows=True,
        arrowstyle="->",
        arrowsize=12
    )

    # Labels: show as {a,b,c}
    labels = {s: "{" + ",".join(map(str, s)) + "}" for s in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.get_current_fig_manager().full_screen_toggle()

    plt.title("Layered Subset Graph", fontsize=10)
    plt.axis("off")
    plt.show()


# Example usage
N = 30
layers = generate_layers(
    n=N,
    max_k=int(math.sqrt(N)),
    base_prob=0.3,
    decay=0.85,
    new_elem_prob=0.1,
    bad_spawn_prob=.05)
# print(layers)
# visualize_layers(layers)
# 
# for i in range(1000):
#     layers = generate_layers(
#         n=N,
#         max_k=int(math.sqrt(N)),
#         base_prob=0.3,
#         decay=0.85,
#         new_elem_prob=0.1,
#         bad_spawn_prob=.05)
#     with open(f"/home/alex/Documents/Mutans Optimization/trees/t{i}.pkl", "wb") as f:
#         pickle.dump(layers, f)
with open("/home/alex/Documents/Mutans Optimization/sample_genes.pkl", "wb") as f:
    pickle.dump(layers, f)
# visualize_layers(layers)
# print(layers)
