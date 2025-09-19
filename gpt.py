import heapq, random
import pickle
from typing import List, Tuple, Dict, Iterable, Optional
from generate_subsets import visualize_layers

def _prepare_scores(singletons: List[Tuple[int]], scores: Optional[Dict[int, float]] = None):
    elems = [s[0] for s in singletons]
    if scores is None:
        # default: equal weights with tiny random jitter to avoid huge ties
        scores = {e: 1.0 + random.random()*1e-6 for e in elems}
    # sort descending by score; keep aligned lists
    sorted_elems = sorted(elems, key=lambda x: -scores.get(x, 0.0))
    sorted_scores = [scores[e] for e in sorted_elems]
    return sorted_elems, sorted_scores, scores

def top_k_pairs(singletons: List[Tuple[int]], 
                K: int, 
                scores: Optional[Dict[int, float]] = None, 
                explore_frac: float = 0.0,
                coverage_decay: float = 0.0) -> List[Tuple[int,int]]:
    """
    Return up to K high-scoring pairs from a singleton layer without enumerating all pairs.
    - scores: optional {elem -> score}; default ~1.0 each
    - explore_frac: add this fraction of K as random extra pairs (exploration)
    - coverage_decay: after selecting a pair (a,b), multiply s[a], s[b] by (1-coverage_decay)
    """
    if K <= 0 or len(singletons) < 2:
        return []
    elems, svals, score_map = _prepare_scores(singletons, scores)
    n = len(elems)

    # Max-heap of (-score, i, j), with j > i
    heap = []
    for i in range(min(n-1, K)):  # only need at most K seeds
        pair_score = svals[i] + svals[i+1]
        heapq.heappush(heap, (-pair_score, i, i+1))
    seen = set((i, i+1) for i in range(min(n-1, K)))
    picked = []

    # local mutable scores for coverage decay
    local_scores = svals[:]

    while heap and len(picked) < K:
        neg_sc, i, j = heapq.heappop(heap)
        a, b = elems[i], elems[j]
        picked.append(tuple(sorted((a, b))))

        if coverage_decay > 0:
            # reweight a and b to encourage coverage
            local_scores[i] *= (1.0 - coverage_decay)
            local_scores[j] *= (1.0 - coverage_decay)

        # push neighbor (i, j+1)
        if j + 1 < n:
            cand = (i, j+1)
            if cand not in seen:
                seen.add(cand)
                heapq.heappush(heap, (-(local_scores[i] + local_scores[j+1]), i, j+1))

    # Optional exploration: add a small random sample of remaining pairs
    if explore_frac > 0 and len(picked) < K:
        budget = min(int(K * explore_frac), (n*(n-1))//2 - len(picked))
        # sample without enumerating all pairs: try random indices
        added = 0
        have = set(picked)
        attempts = 0
        max_attempts = 50 * max(1, budget)
        while added < budget and attempts < max_attempts:
            i = random.randrange(0, n-1)
            j = random.randrange(i+1, n)
            pair = tuple(sorted((elems[i], elems[j])))
            if pair not in have:
                picked.append(pair)
                have.add(pair)
                added += 1
            attempts += 1

    # dedup & cap
    picked = list(dict.fromkeys(picked))[:K]
    return picked

def compare_to_actual(candidates, actual):
    """Mark candidates as 1 if they exist in actual_doubles, else 0."""
    actual_set = {tuple(sorted(d)) for d in actual}
    results = [(tup, 1 if tup in actual_set else 0) for tup in candidates]
    results.sort(key=lambda x: x[1], reverse=True)
    return results
with open("sample_genes.pkl", "rb") as f:
    actual = pickle.load(f)
singletons = actual[1]
print(singletons)
scores = None
pairs  = top_k_pairs(singletons, K=200, scores=scores, explore_frac=0.5, coverage_decay=0.5)

print(type(actual) )
