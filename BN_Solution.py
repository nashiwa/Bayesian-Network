import itertools

class Node:
    def __init__(self, name, states=[0,1]):
        self.name = name
        self.parents = []   # List of parent nodes
        self.children = []  # List of child nodes
        self.cpd = {}       # Conditional probability distribution
        self.states = states  # Possible values the node can take

    def add_parent(self, node):
        """Link another node as a parent of this node."""
        self.parents.append(node)
        node.add_child(self)

    def add_child(self, node):
        """Link another node as a child of this node."""
        self.children.append(node)

    def set_cpd(self, cpd):
        """Assign a conditional probability distribution to the node."""
        self.cpd = cpd

    def get_prob(self, value, evidence, network):
        """Return probability of a given value based on provided evidence."""
        
        # If the node has parent nodes
        if self.parents:
            parent_values = []
            
            for parent in self.parents:
                if parent.name in evidence:  # Parent's value is provided in the evidence
                    parent_values.append(parent.name + '=' + str(evidence[parent.name]))
                else:  # Estimate the most probable state for the parent based on other evidence
                    parent_probs = network.query(parent.name, evidence)
                    most_likely_state = max(parent_probs, key=parent_probs.get)
                    parent_values.append(parent.name + '=' + str(most_likely_state))
            
            key = ','.join(parent_values)
            return self.cpd[key][value]
        else:  
            # Node does not have parents, so we return its marginal probability
            return self.cpd[value]


class BayesianNetwork:
    def __init__(self, nodes={}):
        self.nodes = nodes  # Nodes present in the network

    def get_node(self, name):
        """Retrieve a node by its name from the network."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None  # Return None if node is not found

    def query(self, query, evidence={}):
        """Calculate the probability distribution of a node based on given evidence."""
        node = self.get_node(query)
        
        if not node:
            return None  # Node not found in the network

        # If the node does not have parents or all its parents have known values from evidence
        if not node.parents or all(parent.name in evidence for parent in node.parents):
            return {state: node.get_prob(state, evidence, self) for state in node.states}
        else:
            total_prob = {state: 0 for state in node.states}

            # List all possible combinations of parent states
            parent_states_combinations = list(itertools.product(*[parent.states for parent in node.parents]))

            for state_combo in parent_states_combinations:
                parent_evidence = {parent.name: state for parent, state in zip(node.parents, state_combo)}
                prob_product = 1

                # Calculate joint probability of this particular combination of parent states
                for parent, state in zip(node.parents, state_combo):
                    if parent.name not in evidence:
                        prob_product *= self.query(parent.name, evidence)[state]
                    else:
                        prob_product *= parent.get_prob(state, evidence, self)

                for state in node.states:
                    total_prob[state] += prob_product * node.get_prob(state, {**evidence, **parent_evidence}, self)

            # Adjust the probabilities so they total to 1
            norm_factor = sum(total_prob.values())
            return {state: prob / norm_factor for state, prob in total_prob.items()}
