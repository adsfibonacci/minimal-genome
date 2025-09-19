import pickle
from generate_subsets import visualize_layers

with open("/home/alex/Documents/Mutans Optimization/sample_genes.pkl", "rb") as f:
    layers = pickle.load(f)
singles = layers[1]
visualize_layers(layers)
