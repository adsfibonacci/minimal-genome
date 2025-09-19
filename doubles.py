import pickle
import itertools
from collections import defaultdict 
from generate_subsets import visualize_layers
from operator import itemgetter

all_trees = []
for i in range(999):
    with open(f"/home/alex/Documents/Mutans Optimization/trees/t{i}.pkl", "rb") as f:
        all_trees.append(pickle.load(f))
        pass
    pass
all_singles = [ layers[1] for layers in all_trees ]
all_doubles = [ layers[2] for layers in all_trees ]

counts = defaultdict(int)
# Count how often a pair (a,b) DOES appear in doubles
successes = defaultdict(int)

for singles, doubles in zip(all_singles, all_doubles):
    # flatten singleton tuples like (4,) -> 4
    singleton_vals = [s[0] for s in singles]
    double_pairs = {tuple(sorted(d)) for d in doubles}

    # For each possible pair of singletons
    for a, b in itertools.combinations(singleton_vals, 2):
        pair = tuple(sorted((a, b)))
        counts[pair] += 1
        if pair in double_pairs:
            successes[pair] += 1

# Compute empirical probabilities
probs = {pair: successes[pair] / counts[pair]
         for pair in counts if counts[pair] > 0}

last = []
with open(f"/home/alex/Documents/Mutans Optimization/trees/t{999}.pkl", "rb") as f:
    last = pickle.load(f)
last_singles = last[1]
last_doubles = last[2]

double_probs = { tuple(sorted(combo)) : probs.get(tuple(sorted(combo)), 0.0) for combo in itertools.combinations([l[0] for l in last_singles], 2)}

res = dict(sorted(double_probs.items(), key=itemgetter(1), reverse=True)[:1485])
good_pred = 0
for k,v in res:
    if k in last_doubles:
        good_pred += 1
print(good_pred)
print(len(last_doubles))
print(len(double_probs))

