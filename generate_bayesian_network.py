import random
import itertools
from bayesian_network import Node, BayesianNetwork
from common_function import print_network_structure


def generate_bayesian_network_corrected_v3(node_names, density):
    n = len(node_names)

    # Check if the density is feasible
    if density >= n:
        raise ValueError(
            f"Density too high. Maximum allowable density for {n} nodes is {n-1}.")

    # Create nodes
    nodes = {name: Node(name) for name in node_names}

    # Setting parent-child relationships based on density
    for i, name in enumerate(node_names):
        node = nodes[name]
        # Only consider previous nodes as potential parents
        possible_parents = [nodes[n] for n in node_names[:i]]

        # Limit the number of parents to the maximum possible or specified density
        num_parents = min(len(possible_parents), density)
        selected_parents = random.sample(
            possible_parents, num_parents) if possible_parents else []

        for parent in selected_parents:
            node.add_parent(parent)

    # Automatically generate CPDs
    for node in nodes.values():
        if not node.parents:
            # Node has no parents, assign random marginal probabilities
            node.set_cpd({0: random.uniform(0, 1), 1: random.uniform(0, 1)})
        else:
            # Node has parents, generate CPD for each combination of parent states
            parent_states_combinations = list(
                itertools.product(*[[0, 1] for _ in node.parents]))
            cpd = {}
            for combo in parent_states_combinations:
                parent_values = ','.join(
                    [f'{parent.name}={state}' for parent, state in zip(node.parents, combo)])
                cpd[parent_values] = {0: random.uniform(
                    0, 1), 1: random.uniform(0, 1)}
            node.set_cpd(cpd)

    # Creating the Bayesian Network
    network = BayesianNetwork(list(nodes.values()))
    return network


node_names = ['A', 'B', 'C', 'D']
density = 3
try:
    network = generate_bayesian_network_corrected_v3(node_names, density)
except ValueError as e:
    network = None
    print(e)

if network:
    print_network_structure(network)
    print('\n')
    for node_name in node_names:
        node = network.get_node(node_name)
        print(f'CPD of {node_name}: {node.cpd}\n')
else:
    print('No Network Found')
