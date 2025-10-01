import random, math, pickle

def build_oracle(graph):
    # Convert to set of frozensets for O(1) lookup
    functional = set(frozenset(s) for sets in graph.values() for s in sets)
    return lambda s: frozenset(s) in functional


def mcte(graph, iterations=1000, max_depth=5):
    oracle = build_oracle(graph)

    frontier = [frozenset(s) for s in graph[1]]
    tested = set(frontier)
    candidates = {1: list(frontier)}

    for depth in range(2, max_depth + 1):
        candidates[depth] = []
        for _ in range(iterations):
            # Pick a random functional set from previous layer
            base = random.choice(candidates[depth - 1])
            # Pick a random element not in the set
            new_elem = random.randint(1, max(max(e for s in graph.values() for tup in s for e in tup), 30))
            if new_elem in base:
                continue
            new_set = frozenset(base | {new_elem})
            if new_set not in tested:
                tested.add(new_set)
                if oracle(new_set):
                    candidates[depth].append(set(new_set))
                    pass
                pass
            pass
        pass
    candidates[1] = [set(i) for i in candidates[1]]
    return candidates, tested


N = 5
with open("sample_genes.pkl", "rb") as f:
    graph = pickle.load(f)
candidates, tested = mcte(graph, iterations=500, max_depth=N)
# print(candidates)

# print("Discovered doubles:", [ set(i) for i in candidates[2]] )
# print("Discovered triples:", [ set(i) for i in candidates[3]] )

max_graph = list(graph.keys())[-1]
max_candidate = list(candidates.keys())[-1] if len(candidates[list(candidates.keys())[-1]]) != 0 else list(candidates.keys())[-2]
print(max_graph)
print(max_candidate)
print("True Max Depth: ", [ sorted(list(i)) for i in graph[max_graph]])
print("Candidate Max Depth: ", [ sorted(list(i)) for i in candidates[max_candidate]])
print("Total tested:", len(tested))
# print("True Graph: ", graph)
# print("Candidate Graph: ", candidates)
