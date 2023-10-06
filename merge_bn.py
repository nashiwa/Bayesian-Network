def find_parent(node1, node2):
    """Find which of the two nodes is the parent of the other."""
    if node1 in node2.parents:
        return node1
    elif node2 in node1.parents:
        return node2
    else:
        return None  # Neither node is the parent of the other


def generate_combined_states(grandparents):
    """Generate the combined states based on the number of grandparents with the updated format."""

    # Create a list of lists where each inner list has states in the format 'grandParentName=state'
    states_list = [[f'{gp.name}={state}' for state in gp.states]
                   for gp in grandparents]

    # Generate the Cartesian product of the states
    combined_states = list(itertools.product(*states_list))

    # Convert the tuples to comma-separated strings for easier indexing
    return [','.join(state_tuple) for state_tuple in combined_states]


def calculate_combined_cpd_generic_v3(network, node_names):
    # Ensure that the node names list contains exactly two nodes
    if len(node_names) != 2:
        raise ValueError("Exactly two node names must be provided")

    # Retrieve the nodes from the network using the provided names
    node1 = network.get_node(node_names[0])
    node2 = network.get_node(node_names[1])

    if node1 is None or node2 is None:
        raise ValueError("One or both nodes not found in the network")

    # Find out which node is the parent and which one is the child
    parent_node = find_parent(node1, node2)
    child_node = node2 if parent_node == node1 else node1

    # Get the grandparents (i.e., parents of the parent node)
    grandparents = parent_node.parents

    combined_cpd = {}

    # Generate combined states based on the number of grandparents
    combined_states = generate_combined_states(grandparents)

    for key in combined_states:
        combined_cpd[key] = {}

        # Calculating the joint probability for EF=0 and EF=1
        combined_0 = (child_node.cpd[f'{parent_node.name}=0'][0] * parent_node.cpd[key][0]) + (
            child_node.cpd[f'{parent_node.name}=1'][0] * parent_node.cpd[key][1])
        combined_1 = (child_node.cpd[f'{parent_node.name}=0'][1] * parent_node.cpd[key][0]) + (
            child_node.cpd[f'{parent_node.name}=1'][1] * parent_node.cpd[key][1])

        # Assigning the computed probabilities to the combined CPD
        combined_cpd[key][0] = combined_0
        combined_cpd[key][1] = combined_1

    return combined_cpd


node_names = ['E', 'D']
combined_cpd_EF_v3 = calculate_combined_cpd_generic_v3(network, node_names)

print(combined_cpd_EF_v3)
