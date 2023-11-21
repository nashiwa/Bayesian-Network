# Greedy-Degree method
def print_degree_of_pairs(network):
    degree_pairs = {}
    for node in network.nodes:
        for child in node.children:
            # Calculate combined degree, considering extra degree for the child
            degree = len(node.parents) + len(node.children) + \
                len(child.parents) + len(child.children)
            degree_pairs[(node.name, child.name)] = degree
    return degree_pairs


def find_pair_greedy_degree(network):
    min_degree = float('inf')
    min_degree_pair = (None, None)

    for node in network.nodes:
        for child in node.children:
            # Calculate combined degree
            degree = len(node.parents) + len(node.children) + \
                len(child.parents) + len(child.children)
            if degree < min_degree:
                min_degree = degree
                min_degree_pair = (node.name, child.name)

    return min_degree_pair


# Greedy-Fill-in method
def find_pair_greedy_fill_in(network):
    min_fill_in = float('inf')
    min_fill_in_pair = (None, None)

    for parent in network.nodes:
        for child in parent.children:
            # Calculate fill-ins required to combine parent and child
            fill_in = len(set(parent.parents + parent.children + child.parents + child.children)
                          - set([parent, child])
                          - set(parent.children)
                          - set(child.parents))
            if fill_in < min_fill_in:
                min_fill_in = fill_in
                min_fill_in_pair = (parent.name, child.name)

    return min_fill_in_pair


def print_fill_ins_for_each_pair(network):
    fill_ins_info = {}

    for parent in network.nodes:
        for child in parent.children:
            # Calculate fill-ins required to combine parent and child
            fill_in = len(set(parent.parents + parent.children + child.parents + child.children)
                          - set([parent, child])
                          - set(parent.children)
                          - set(child.parents))

            fill_ins_info[(parent.name, child.name)] = fill_in

    return fill_ins_info
