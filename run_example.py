from bayesian_network import Node, BayesianNetwork

# Function to see a network structure
from common_function import print_network_structure

# Combine two nodes if the parent node has multiple output
from combine_node_multi_outtput import combine_nodes_and_update_network_multi_output

# Combine two nodes if the parent node has single output
from combine_node_single_output import combine_nodes_and_update_network_single_output

# Very basic function to combine two nodes
from combine_node_basic_function import calculate_combined_cpd_generic

# pass a network to find which nodes can be minimized
from find_node_to_minimize import find_pair_greedy_degree, find_pair_greedy_fill_in


# 1. Create Nodes with their possible states
A = Node(name="A")
B = Node(name="B")
C = Node(name="C")
D = Node(name="D")

# 2. Setting parent-child relationships
C.add_parent(A)
C.add_parent(B)
D.add_parent(C)


# 3. Setting CPDs
A.set_cpd({0: 1/3, 1: 2/3})
B.set_cpd({0: 2/5, 1: 3/5})
C.set_cpd({
    'A=0,B=0': {0: 1, 1: 0},
    'A=0,B=1': {0: 1/5, 1: 4/5},
    'A=1,B=0': {0: 1/2, 1: 1/2},
    'A=1,B=1': {0: 1/4, 1: 3/4}
})
D.set_cpd({
    'C=0': {0: 7/8, 1: 1/8},
    'C=1': {0: 1/2, 1: 1/2}
})

# Creating the Bayesian Network
test_network = BayesianNetwork([A, B, C, D])

modified_test_network = combine_nodes_and_update_network_single_output(test_network, 'C', 'D')
print_network_structure(modified_test_network)


nodes_to_minimize = find_pair_greedy_degree(test_network)
print(nodes_to_minimize)
