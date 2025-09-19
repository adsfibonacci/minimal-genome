Genes are treated as numerical values. Removing a collection while maintaining viability creates a new node. 
Give the list of single genes turned off while maintaining viability a new node, and this forms a graph (not necessarily a connected one). 
Other base sets can be added as extra viable node sets, but the singletons are all that is needed. 
The goal is to maximize the number of genes removed from the genome. 

The current iteration is using Monte-Carlo tree traversal methods.

Dummy graphs are probabilistically generated in the `generate_subsets.py` file. 
This file also contains a visualizer for the network of viable removed genomes. 

The file `mcts.py` is the Monte-Carlo simulation file. 

The file `gpt.py` is used as a launching pad code containing anything GPT generated. 
