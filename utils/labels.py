def get_drautz_label(key):
    categories = ['cluster', 'sp2', 'sp3', 'bulk', 'amorph']
    for cat in categories:
        if cat in key:
            return cat
        else:
            continue

