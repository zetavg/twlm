def deep_diff_dict(dict1, dict2, current_path=None):
    if current_path is None:
        current_path = []

    # Find keys unique to dict1 (removed keys)
    unique_keys_in_dict1 = set(dict1.keys()) - set(dict2.keys())

    # Find keys unique to dict2 (added keys)
    unique_keys_in_dict2 = set(dict2.keys()) - set(dict1.keys())

    # Find common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())

    # Initialize containers for added, removed, and updated keys
    added_values = {}
    removed_values = {}
    updated_values = {}

    # Check removed keys (in dict1 but not in dict2)
    for key in unique_keys_in_dict1:
        removed_values[".".join(current_path + [key])] = dict1[key]

    # Check added keys (in dict2 but not in dict1)
    for key in unique_keys_in_dict2:
        added_values[".".join(current_path + [key])] = dict2[key]

    # Check keys present in both dictionaries
    for key in common_keys:
        new_path = current_path + [key]
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            # Perform a deep diff if both values are dictionaries
            nested_diff = deep_diff_dict(dict1[key], dict2[key], new_path)

            added_values.update(nested_diff['added'])
            removed_values.update(nested_diff['removed'])
            updated_values.update(nested_diff['updated'])
        elif dict1[key] != dict2[key]:
            # If the values are not dictionaries and differ, mark them as updated
            updated_values[".".join(new_path)] = (dict1[key], dict2[key])

    return {
        'added': added_values,
        'updated': updated_values,
        'removed': removed_values,
    }


# def deep_diff_dict(dict1, dict2):
#     # Find keys unique to dict1 (removed keys)
#     unique_keys_in_dict1 = set(dict1.keys()) - set(dict2.keys())

#     # Find keys unique to dict2 (added keys)
#     unique_keys_in_dict2 = set(dict2.keys()) - set(dict1.keys())

#     # Find common keys
#     common_keys = set(dict1.keys()) & set(dict2.keys())

#     # Initialize containers for added, removed, and updated keys
#     added_values = {}
#     removed_values = {}
#     updated_values = {}

#     # Check removed keys (in dict1 but not in dict2)
#     for key in unique_keys_in_dict1:
#         removed_values[key] = dict1[key]

#     # Check added keys (in dict2 but not in dict1)
#     for key in unique_keys_in_dict2:
#         added_values[key] = dict2[key]

#     # Check keys present in both dictionaries
#     for key in common_keys:
#         if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
#             # Perform a deep diff if both values are dictionaries
#             nested_diff = deep_diff_dict(dict1[key], dict2[key])
#             if nested_diff != {'added': {}, 'updated': {}, 'removed': {}}:
#                 updated_values[key] = nested_diff
#         elif dict1[key] != dict2[key]:
#             # If the values are not dictionaries and differ, mark them as updated
#             updated_values[key] = (dict1[key], dict2[key])

#     return {
#         'added': added_values,
#         'updated': updated_values,
#         'removed': removed_values,
#     }
