import itertools
from common_function import find_grandparents, generate_combined_states
from bayesian_network import Node


def calculate_combined_cpd_no_grnadparent_multiple_parents(network, child_node_name):
    # Retrieve the child node from the network using the provided name
    child_node = network.get_node(child_node_name)

    # Retrieve all parents of the child node
    parents = child_node.parents

    # Generate all possible combinations of parent states
    parent_states_combinations = list(
        itertools.product(*[parent.states for parent in parents]))

    # Initialize the combined CPD
    combined_cpd = {0: 0, 1: 0}

    # Iterate over all possible combinations of parent states
    for state_combo in parent_states_combinations:
        # Construct the key for the CPD of the child node
        key = ','.join(f"{parents[i].name}={state}" for i,
                       state in enumerate(state_combo))

        # Compute the joint probability of this parent state combination
        joint_prob = 1
        for parent, state in zip(parents, state_combo):
            joint_prob *= parent.cpd[state]

        # Update the combined CPD for each state of the child node
        for state in child_node.states:
            combined_cpd[state] += joint_prob * child_node.cpd[key][state]

    return combined_cpd


def calculate_combined_cpd_with_multiple_grandparents_not_connected_parents(network, child_node_name):
    child_node = network.get_node(child_node_name)

    # Find all unique grandparents
    grandparents = find_grandparents(child_node)

    # Generate combined states based on the number of grandparents
    combined_states = generate_combined_states(grandparents)

    combined_cpd = {}

    # Iterate through each combined grandparent state
    for combined_state in combined_states:
        combined_cpd[combined_state] = {}

        # Split the combined state into individual grandparent states
        gp_state_dict = {state.split('=')[0]: int(
            state.split('=')[1]) for state in combined_state.split(',')}

        # Calculate the probability for each state of the child node
        for state in child_node.states:
            prob = 0
            for parent_state_combination in itertools.product(*[parent.states for parent in child_node.parents]):
                # Create a dictionary for the parent states
                parent_state_dict = {parent.name: parent_state for parent, parent_state in zip(
                    child_node.parents, parent_state_combination)}

                # Calculate the joint probability of the parent states
                parent_prob_product = 1
                for parent in child_node.parents:
                    if parent.parents:
                        # Construct the CPD key for the parent considering all of its parents
                        parent_cpd_key = ','.join(
                            [f'{gp.name}={gp_state_dict[gp.name]}' for gp in parent.parents])
                    else:
                        # If the parent has no grandparents and no CPD dictionary, use its marginal probability
                        parent_prob_product *= parent.cpd[parent_state_dict[parent.name]]
                        continue

                    parent_prob_product *= parent.cpd[parent_cpd_key][parent_state_dict[parent.name]]

                # Update the probability for the child state
                child_cpd_key = ','.join(
                    [f'{parent_name}={parent_state}' for parent_name, parent_state in parent_state_dict.items()])
                prob += parent_prob_product * \
                    child_node.cpd[child_cpd_key][state]

            # Assign the calculated probability to the combined CPD
            combined_cpd[combined_state][state] = prob

    # Normalize the probabilities for each combined grandparent state
    for combined_state in combined_cpd:
        total_prob = sum(combined_cpd[combined_state].values())
        for state in combined_cpd[combined_state]:
            combined_cpd[combined_state][state] /= total_prob

    return combined_cpd


def combine_nodes_and_update_network_single_output(network, parent_node_name, child_node_name):
    child_node = network.get_node(child_node_name)
    grandparents = find_grandparents(child_node)
    parent_str = ''
    for parent in child_node.parents:
        parent_str += parent.name

    # If there are no grandparents, calculate the CPD directly
    if not grandparents:
        combined_cpd = calculate_combined_cpd_no_grnadparent_multiple_parents(
            network, child_node_name)
        combined_node_name = child_node_name + parent_str
        print(f'Combined CPD of {combined_node_name}: {combined_cpd}')
    else:
        combined_cpd = calculate_combined_cpd_with_multiple_grandparents_not_connected_parents(
            network, child_node_name)
        combined_node_name = child_node_name + parent_str
        print(f'Combined CPD of {combined_node_name}: {combined_cpd}')

    # Create a new combined node
    combined_node = Node(name=combined_node_name)
    combined_node.set_cpd(combined_cpd)

    for parent in child_node.parents:
        for grandparent in parent.parents:
            combined_node.parents.append(grandparent)
            grandparent.children.append(combined_node)
            grandparent.children.remove(parent)
        network.nodes.remove(parent)

    # Remove the original parent and child nodes from the network
    network.nodes.remove(child_node)

    # Add the combined node to the network
    network.nodes.append(combined_node)

    # If the child node has any children, update their parent to the combined node
    for child in child_node.children:
        child.parents = [combined_node]
        combined_node.children.append(child)

    return network
