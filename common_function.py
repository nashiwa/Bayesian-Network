import itertools

def print_network_structure(network):
    for node in network.nodes:
        # Fetching the names of the parent and child nodes
        parent_names = [parent.name for parent in node.parents]
        child_names = [child.name for child in node.children]

        # Formatting the output
        parents_str = ', '.join(parent_names) if parent_names else "No parents"
        children_str = ', '.join(child_names) if child_names else "No children"

        # Printing the details for each node
        print(f"Node '{node.name}': Parents -> [{parents_str}], Children -> [{children_str}]")

def find_grandparents(child_node):
    """Find all unique grandparents (parents of parents) of the given child node."""
    grandparents = []
    for parent in child_node.parents:
        for grandparent in parent.parents:
            if grandparent not in grandparents:
                grandparents.append(grandparent)
    return grandparents


def generate_combined_states(grandparents):
    """Generate the combined states based on the number of grandparents with the updated format."""

    # Create a list of lists where each inner list has states in the format 'grandParentName=state'
    states_list = [[f'{gp.name}={state}' for state in gp.states]
                   for gp in grandparents]

    # Generate the Cartesian product of the states
    combined_states = list(itertools.product(*states_list))

    # Convert the tuples to comma-separated strings for easier indexing
    return [','.join(state_tuple) for state_tuple in combined_states]