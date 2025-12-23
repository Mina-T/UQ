def get_drautz_label(key):
    '''
    given a name, returns Drautz category and category-id
    '''
    categories_dict = {'cluster':1, 'sp2':2, 'sp3':3, 'bulk':4, 'amorph':5}
    for cat in categories_dict.keys():
        if cat in key:
            return cat, categories_dict[cat]
    
    return None, None 


def drautz_atom_label(cat, sys_id, atom_id): # make it drautz independent
    '''
    labels each atom of Drautz test set. 
    '''
    cat_id = get_drautz_label(cat)
    atom_label = f"{cat_id}_{sys_id}_{atom_id}"
    return atom_label

