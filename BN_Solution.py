class Node:
    def __init__(self, name, states):
        self.name = name
        self.parents = []
        self.children = []
        self.cpd = {}
        self.states = states  # all possible states of the node

    def add_parent(self, node):
        self.parents.append(node)
        node.add_child(self)

    def add_child(self, node):
        self.children.append(node)

    def set_cpd(self, cpd):
        self.cpd = cpd

    def get_prob(self, value, evidence, network):
        if self.parents:
            parent_values = []
            for parent in self.parents:
                if parent.name in evidence:
                    parent_values.append(parent.name + '=' + str(evidence[parent.name]))
                else:
                    parent_probs = network.query(parent.name, evidence)
                    most_likely_state = max(parent_probs, key=parent_probs.get)
                    parent_values.append(parent.name + '=' + str(most_likely_state))

            key = ','.join(parent_values)
            return self.cpd[key][value]
        else:
            return self.cpd[value]


class BayesianNetwork:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def query(self, query, evidence):
        node = self.get_node(query)
        if not node:
            return None  # Node not found in the network

        # If the node has no parents or all its parents are in evidence
        if not node.parents or all(parent.name in evidence for parent in node.parents):
            return {state: node.get_prob(state, evidence, self) for state in node.states}
        else:
            total_prob = {state: 0 for state in node.states}

            # Generate all possible combinations of states for the parents
            import itertools
            parent_states_combinations = list(itertools.product(*[parent.states for parent in node.parents]))

            for state_combo in parent_states_combinations:
                parent_evidence = {parent.name: state for parent, state in zip(node.parents, state_combo)}
                prob_product = 1

                for parent, state in zip(node.parents, state_combo):
                    if parent.name not in evidence:
                        prob_product *= self.query(parent.name, evidence)[state]
                    else:
                        prob_product *= parent.get_prob(state, evidence, self)

                for state in node.states:
                    total_prob[state] += prob_product * node.get_prob(state, {**evidence, **parent_evidence}, self)

            # Normalize the probabilities so they sum to 1
            norm_factor = sum(total_prob.values())
            return {state: prob / norm_factor for state, prob in total_prob.items()}
