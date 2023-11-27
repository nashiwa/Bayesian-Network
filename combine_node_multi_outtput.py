import itertools
from common_function import generate_combined_states
from bayesian_network import Node


def calculate_combined_cpd_generic_multi_output(network, parent_node_name, child_node_name):
    # Retrieve the parent and child nodes from the network
    parent_node = network.get_node(parent_node_name)
    child_node = network.get_node(child_node_name)

    # Find grandparents (parents of the parent node)
    grandparents = parent_node.parents

    combined_cpd = {}

    if grandparents:
        # Generate combined states based on the number of grandparents
        combined_states = generate_combined_states(grandparents)

        for gp_state in combined_states:
            combined_cpd[gp_state] = {}

            for parent_state in parent_node.states:
                for child_state in child_node.states:
                    # Generating keys for CPD lookups
                    cpd_key_parent = f'{gp_state},{parent_node.name}={parent_state}'
                    cpd_key_child = f'{parent_node.name}={parent_state}'

                    # Calculating joint probability
                    joint_prob = child_node.cpd[cpd_key_child][child_state] * \
                        parent_node.cpd[gp_state][parent_state]

                    # Defining a unique key for each state combination
                    combined_state_key = f'{child_node.name}={child_state},{parent_node.name}={parent_state}'

                    # Assigning the computed probability
                    combined_cpd[gp_state][combined_state_key] = joint_prob
    else:
        # Handle case with no grandparents
        for parent_state in parent_node.states:
            for child_state in child_node.states:
                cpd_key_parent = f'{parent_node.name}={parent_state}'
                cpd_key_child = f'{parent_node.name}={parent_state}'

                joint_prob = child_node.cpd[cpd_key_child][child_state] * \
                    parent_node.cpd[parent_state]

                combined_state_key = f'{child_node.name}={child_state},{parent_node.name}={parent_state}'

                combined_cpd[combined_state_key] = joint_prob
    return combined_cpd


def calculate_combined_cpd_generic_multi_output_pre_combined_parent(network, parent_node_name, child_node_name):
    parent_node = network.get_node(parent_node_name)
    child_node = network.get_node(child_node_name)

    # Extract the original parent name from the child's CPD keys
    original_parent_of_child = next(iter(child_node.cpd)).split('=')[0]

    # Extract the previously combined child name from the new comboned node
    previouly_combined_child = parent_node_name.replace(
        original_parent_of_child, '')

    # Handle the case where the parent node has multiple or no grandparents
    grandparents = parent_node.parents
    combined_cpd = {}

    if grandparents:
        for gp_combination in itertools.product(*[gp.states for gp in grandparents]):
            gp_state_key = ','.join(
                [f'{gp.name}={state}' for gp, state in zip(grandparents, gp_combination)])
            combined_cpd[gp_state_key] = {}

            for c_state in [0, 1]:
                for d_state in child_node.states:
                    joint_prob = 0
                    for a_val in [0, 1]:
                        cpd_key_child = f'{original_parent_of_child}={a_val}'
                        combined_state_key_for_cpd = f'{previouly_combined_child}={c_state},{original_parent_of_child}={a_val}'

                        if cpd_key_child in child_node.cpd and combined_state_key_for_cpd in parent_node.cpd[gp_state_key]:
                            joint_prob += child_node.cpd[cpd_key_child][d_state] * \
                                parent_node.cpd[gp_state_key][combined_state_key_for_cpd]

                    combined_state_key = f'{child_node.name}={d_state},{previouly_combined_child}={c_state}'
                    combined_cpd[gp_state_key][combined_state_key] = joint_prob
    else:
        # If there are no grandparents, the CPD will only depend on the parent node's states
        for c_state in [0, 1]:
            for d_state in child_node.states:
                joint_prob = 0
                for a_val in [0, 1]:
                    cpd_key_child = f'{original_parent_of_child}={a_val}'
                    combined_state_key_for_cpd = f'{previouly_combined_child}={c_state},{original_parent_of_child}={a_val}'

                    if cpd_key_child in child_node.cpd:
                        joint_prob += child_node.cpd[cpd_key_child][d_state] * \
                            parent_node.states[c_state]

                combined_state_key = f'{child_node.name}={d_state},{previouly_combined_child}={c_state}'
                combined_cpd[combined_state_key] = joint_prob
    return combined_cpd


def combine_nodes_and_update_network_multi_output(network, parent_node_name, child_node_name):
    parent_node = network.get_node(parent_node_name)
    child_node = network.get_node(child_node_name)

    if len(parent_node.states) == 2:
        combined_cpd = calculate_combined_cpd_generic_multi_output(
            network, parent_node_name, child_node_name)
        combined_node_name = child_node_name + parent_node_name
        print(f'Combined CPD of {combined_node_name}: {combined_cpd}')
    else:
        combined_cpd = calculate_combined_cpd_generic_multi_output_pre_combined_parent(
            network, parent_node_name, child_node_name)
        # Extract the original parent name from the child's CPD keys
        original_parent_of_child = next(iter(child_node.cpd)).split('=')[0]
        # Extract the previously combined child name from the new comboned node
        previouly_combined_child = parent_node_name.replace(
            original_parent_of_child, '')
        combined_node_name = child_node_name + previouly_combined_child
        print(f'Combined CPD of {combined_node_name}: {combined_cpd}')

    # Create a new combined node
    combined_node = Node(name=combined_node_name, states=[
                         (0, 0), (1, 0), (0, 1), (1, 1)])
    combined_node.set_cpd(combined_cpd)

    # Set the parents of the combined node to be the parents of the original parent node
    combined_node.parents = parent_node.parents

    # For each parent of the original parent node, add the combined node as a child
    for grandparent in parent_node.parents:
        grandparent.children.append(combined_node)
        grandparent.children.remove(parent_node)

    # Remove the original parent and child nodes from the network
    network.nodes.remove(parent_node)
    network.nodes.remove(child_node)

    # Add the combined node to the network
    network.nodes.append(combined_node)

    # Update the child nodes of the original parent node
    for child in parent_node.children:
        if child != child_node:
            child.parents = [
                combined_node] if combined_node not in child.parents else child.parents
            combined_node.children.append(child)

    # If the child node has any children, update their parent to the combined node
    for child in child_node.children:
        child.parents = [combined_node]
        combined_node.children.append(child)

    return network
