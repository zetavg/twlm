def shallow_diff_dict(dict1, dict2):
    # Find keys unique to dict1 (removed keys)
    unique_keys_in_dict1 = set(dict1.keys()) - set(dict2.keys())

    # Find keys unique to dict2 (added keys)
    unique_keys_in_dict2 = set(dict2.keys()) - set(dict1.keys())

    # Find common keys with different values (updated keys)
    common_keys = set(dict1.keys()) & set(dict2.keys())
    updated_values = {
        key: (dict1[key], dict2[key])
        for key in common_keys if dict1[key] != dict2[key]}

    # Find removed values
    removed_values = {key: dict1[key] for key in unique_keys_in_dict1}

    # Find added values
    added_values = {key: dict2[key] for key in unique_keys_in_dict2}

    return {
        'added': added_values,
        'updated': updated_values,
        'removed': removed_values,
    }
