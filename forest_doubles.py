import pickle
import numpy as np
import itertools
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Step 1: Load all trees
# ---------------------------
all_trees = []
for i in range(999):
    with open(f"/home/alex/Documents/Mutans Optimization/trees/t{i}.pkl", "rb") as f:
        all_trees.append(pickle.load(f))

# Determine the ground set size
N = max(max(s[0] for layers in all_trees for s in layers[1]), 
        max(max(d) for layers in all_trees for d in layers[2]))

# ---------------------------
# Step 2: Build dataset
# ---------------------------
def build_dataset(all_trees, n):
    pair_list = list(itertools.combinations(range(n+1), 2))
    num_pairs = len(pair_list)
    num_trees = len(all_trees)
    
    X = np.zeros((num_trees, n+1), dtype=int)
    Y = np.zeros((num_trees, num_pairs), dtype=int)
    
    pair_idx = {pair: i for i, pair in enumerate(pair_list)}
    
    for i, tree in enumerate(all_trees):
        singles = [s[0] for s in tree.get(1,[])]
        doubles = [tuple(sorted(d)) for d in tree.get(2,[])]
        
        X[i, singles] = 1  # mark which singletons are present
        
        for d in doubles:
            if d in pair_idx:
                Y[i, pair_idx[d]] = 1
    
    return X, Y, pair_list

X, Y, pair_list = build_dataset(all_trees, N)

# ---------------------------
# Step 3: Train classifier
# ---------------------------
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X, Y)

# ---------------------------
# Step 4: Load last tree for prediction
# ---------------------------
with open(f"/home/alex/Documents/Mutans Optimization/trees/t999.pkl", "rb") as f:
    last_tree = pickle.load(f)

last_singles = last_tree[1]

# ---------------------------
# Step 5: Predict doubles
# ---------------------------
def predict_doubles_ml(singletons, clf, pair_list, N, threshold=0.3):
    x = np.zeros((1, N+1), dtype=int)
    x[0, [s[0] for s in singletons]] = 1
    y_prob_list = clf.predict_proba(x)  # list of arrays, one per pair

    preds = []
    for i, pair in enumerate(pair_list):
        probs = y_prob_list[i][0]  # shape may be (1,) or (2,)
        # check if second class exists
        if len(probs) == 2:
            prob_1 = probs[1]
        else:
            # only one class present in training
            # if class 0 only => prob_1 = 0, class 1 only => prob_1 = 1
            prob_1 = 1.0 if clf.classes_[i][0] == 1 else 0.0
        if prob_1 >= threshold:
            preds.append(pair)
    return preds
default_prob = 0.02  # small chance for unseen pairs

for a, b in itertools.combinations([s[0] for s in last_singles], 2):
    pair = (a,b)
    if pair in pair_list:
        prob_1 = get_predicted_prob(pair)  # as before
    else:
        prob_1 = default_prob
    if prob_1 >= threshold:
        preds.append(pair)

predicted_doubles = predict_doubles_ml(last_singles, clf, pair_list, N, threshold=0.03)

print("Predicted doubles for last singleton layer:", predicted_doubles)
