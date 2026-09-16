from collections import OrderedDict


def _is_valid_child_for_parent(parent, child, ntaxa):
    parent_clade = parent[ntaxa:]
    child_size = child.count('1')
    parent_size = parent_clade.count('1')

    if child_size == 0 or child_size >= parent_size:
        return False

    return all(parent_clade[i] == '1' for i, bit in enumerate(child) if bit == '1')


def filter_subsplit_support_for_sampling(taxa, subsplit_supp_dict):
    ntaxa = len(taxa)
    filtered = OrderedDict()

    for parent, child_dict in subsplit_supp_dict.items():
        valid_children = OrderedDict(
            (child, weight)
            for child, weight in child_dict.items()
            if _is_valid_child_for_parent(parent, child, ntaxa)
        )
        if valid_children:
            filtered[parent] = valid_children

    return filtered