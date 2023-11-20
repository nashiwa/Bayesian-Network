import itertools
from common_function import find_grandparents

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
    node = network.get_node(child_node_name)
    if not node.parents:
        return node.cpd

    # Identifying all the grandparents (parents of the node's parents)
    grandparents = set()
    for parent in node.parents:
        grandparents.update(parent.parents)

    # Create a list of all possible combinations of states for the grandparents
    gp_combinations = list(itertools.product(
        *[gp.states for gp in grandparents]))

    # Create a dictionary to hold the combined CPD with keys for each grandparent state combination
    combined_cpd = {
        ",".join(f"{gp.name}={state}" for gp, state in zip(grandparents, gp_combination)): {0: 0, 1: 0}
        for gp_combination in gp_combinations
    }

    # Iterate through each combination of grandparent states
    for gp_combination in gp_combinations:
        gp_state_dict = {gp.name: state for gp,
                         state in zip(grandparents, gp_combination)}

        # Create a dictionary for the combined parent states
        combined_parent_cpd = {}
        for parent in node.parents:
            combined_parent_cpd[parent.name] = {}
            for parent_state in parent.states:
                # Calculate the probability of the parent being in the current state, given the grandparent states
                if parent.parents:
                    # If the parent has grandparents, calculate the conditional probability
                    prob = 0
                    for grandparent in parent.parents:
                        prob += parent.cpd[f"{grandparent.name}={gp_state_dict[grandparent.name]}"][parent_state]
                    prob /= len(parent.parents)
                else:
                    # If the parent has no grandparents, use its marginal probability
                    prob = parent.cpd[parent_state]
                combined_parent_cpd[parent.name][parent_state] = prob

        # Calculate the probability of the node for each of its states
        for state in node.states:
            # Calculate the probability of the node being in the current state, given the parent states
            prob = 0
            for parent_combination in itertools.product(*[parent.states for parent in node.parents]):
                parent_comb_key = ",".join(f"{node.parents[i].name}={parent_combination[i]}"
                                           for i in range(len(node.parents)))
                prob_comb = node.cpd[parent_comb_key][state]
                for i, parent_state in enumerate(parent_combination):
                    prob_comb *= combined_parent_cpd[node.parents[i].name][parent_state]
                prob += prob_comb

            # Assign the calculated probability to the combined CPD
            combined_cpd_key = ",".join(
                f"{gp.name}={gp_state_dict[gp.name]}" for gp in grandparents)
            combined_cpd[combined_cpd_key][state] = prob

        # Normalize the probabilities for the current grandparent state combination
        total_prob = sum(combined_cpd[combined_cpd_key].values())
        for state in node.states:
            combined_cpd[combined_cpd_key][state] /= total_prob

    return combined_cpd


def combine_nodes_single_output(network, parent_node_name, child_node_name):
    child_node = network.get_node(child_node_name)
    grandparents = find_grandparents(child_node)

    # If there are no grandparents, calculate the CPD directly
    if not grandparents:
        print('no grandparents')
        return calculate_combined_cpd_no_grnadparent_multiple_parents(network, child_node_name)
    else:
        print('grandparents')
        return calculate_combined_cpd_with_multiple_grandparents_not_connected_parents(network, child_node_name)
