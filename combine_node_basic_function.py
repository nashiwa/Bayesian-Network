from common_function import find_grandparents, generate_combined_states


def calculate_combined_cpd_generic(network, parent_node, child_node):
    # Retrieve the nodes from the network using the provided names
    parent_node = network.get_node(parent_node)
    child_node = network.get_node(child_node)

    # Get the grandparents (i.e., parents of the parent node)
    # grandparents = parent_node.parents
    grandparents = find_grandparents(child_node)

    combined_cpd = {}

    # If there are no grandparents, calculate the CPD directly
    if not grandparents:
        combined_0 = (child_node.cpd[f'{parent_node.name}=0'][0] * parent_node.cpd[0]) + (
            child_node.cpd[f'{parent_node.name}=1'][0] * parent_node.cpd[1])
        combined_1 = (child_node.cpd[f'{parent_node.name}=0'][1] * parent_node.cpd[0]) + (
            child_node.cpd[f'{parent_node.name}=1'][1] * parent_node.cpd[1])
        combined_cpd = {0: combined_0, 1: combined_1}
    else:
        # Generate combined states based on the number of grandparents
        combined_states = generate_combined_states(grandparents)

        for key in combined_states:
            combined_cpd[key] = {}

            # Calculating the joint probability
            combined_0 = (child_node.cpd[f'{parent_node.name}=0'][0] * parent_node.cpd[key][0]) + (
                child_node.cpd[f'{parent_node.name}=1'][0] * parent_node.cpd[key][1])
            combined_1 = (child_node.cpd[f'{parent_node.name}=0'][1] * parent_node.cpd[key][0]) + (
                child_node.cpd[f'{parent_node.name}=1'][1] * parent_node.cpd[key][1])

            # Assigning the computed probabilities to the combined CPD
            combined_cpd[key][0] = combined_0
            combined_cpd[key][1] = combined_1

    return combined_cpd
