import copy

class Tree(object):
    """
    Generic tree structure.
    Nodes are uniquely identified by an index (idx) into a list
    of the tree node data.
    """
    def __init__(self):
        self.nodes = []

        # maps index of parent nodes to index of child nodes in the
        # nodes list
        self.p_c_edges = dict()

    def reset(self):
        self.nodes = []
        self.p_c_edges = dict()

    def num_nodes(self):
        return len(self.nodes)

    def get_children(self, idx):
        """
        Returns the index of the children nodes.
        """
        return self.p_c_edges[idx]

    def insert_node(self, node_data, parent_idx):
        """
        Inserts a new node that contains node_data with a parent
        at parent_idx.
        If parent_idx is None, then it will be denoted as a root node.
        This will return the idx of the new node.
        """
        new_idx = len(self.nodes)
        self.nodes.append(node_data)
        if parent_idx is not None:
            assert(parent_idx < new_idx)
            self.p_c_edges[parent_idx].append(new_idx)
        self.p_c_edges[new_idx] = []
        return new_idx

    def update_node_data(self, node_idx, node_data):
        """
        Replaces the data stored at node node_idx with new node_data.
        """
        self.nodes[node_idx] = node_data

    def get_node_data(self, node_idx):
        """
        Get the node_data at node_idx
        """
        return self.nodes[node_idx]

    def rebase(self, root_idx):
        """
        Rebases the tree so root_idx is the new root node.
        Any unconnected nodes will be deleted. This invalidates 
        all existing node indices.
        """
        subtree_nodes = sorted(set(self.find_subtree_nodes(root_idx)))
        new_nodes = []
        new_p_c_edges = dict()
        old_to_new_mapping = dict()
        for node_idx in subtree_nodes:
            old_to_new_mapping[node_idx] = len(new_nodes)
            new_nodes.append(self.nodes[node_idx])
        for node_idx in subtree_nodes:
            parent_idx = old_to_new_mapping[node_idx]
            children_indices = []
            for child_idx in self.get_children(node_idx):
                children_indices.append(old_to_new_mapping[child_idx])
            new_p_c_edges[parent_idx] = children_indices
        self.nodes = new_nodes
        self.p_c_edges = new_p_c_edges

    def find_subtree_nodes(self, idx, return_root=True):
        children = self.get_children(idx)
        subtree_list = copy.deepcopy(children)
        for child in children:
            subtree_list += self.find_subtree_nodes(child, False)
        if return_root:
            subtree_list = [idx] + subtree_list
        return subtree_list



if __name__ == "__main__":
    tree = Tree()
    tree.insert_node(0, None)
    tree.insert_node(0, 0)
    tree.insert_node(0, 0)
    tree.insert_node(0, 1)
    tree.insert_node(0, 2)
    tree.insert_node(0, 3)

    subtree_list = tree.find_subtree_nodes(1)
    print(subtree_list)
