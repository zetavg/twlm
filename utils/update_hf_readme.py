import re
import yaml
from huggingface_hub import HfApi, HfFileSystem
from termcolor import colored


def update_hf_readme(path, content, frontmatter={}):
    fs = HfFileSystem()
    readme_file_path = f"{path}/README.md"
    readme_file_content = ''
    if fs.isfile(readme_file_path):
        old_readme_file_content: str = fs.read_text(readme_file_path)  # type: ignore
        old_readme_file_content_split = old_readme_file_content.split('\n---')
        if len(old_readme_file_content_split) > 0:
            frontmatter_yaml_str = old_readme_file_content_split[0]
            try:
                frontmatter_yaml_str = re.sub('^---\n', '', frontmatter_yaml_str)
                old_frontmatter = yaml.safe_load(frontmatter_yaml_str)
                frontmatter = deep_merge(old_frontmatter, frontmatter)
            except Exception as e:
                print(colored(
                    f"Error on reading frontmatter from existing file: {e}.",
                    'red',
                    attrs=['bold']
                ))
                print('frontmatter_yaml_str:', frontmatter_yaml_str)
                print(colored(
                    f"Old frontmatter might be lost.",
                    'red',
                    attrs=['bold']
                ))

    if frontmatter:
        frontmatter_string = yaml.dump(frontmatter, default_flow_style=False)
        readme_file_content += '---\n' + frontmatter_string + '\n---\n'

    readme_file_content += content.strip()

    with fs.open(readme_file_path, 'w') as f:
        f.write(readme_file_content)


def deep_merge(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged:
            if isinstance(merged[key], dict) and isinstance(value, dict):
                # If the key exists in both dictionaries and the values are dictionaries,
                # recursively merge the inner dictionaries
                merged[key] = deep_merge(merged[key], value)
            elif isinstance(merged[key], list):
                new_value = value
                if not isinstance(new_value, list):
                    new_value = [new_value]
                # Add new values to the list, if they don't already exist
                for v in new_value:
                    if v not in merged[key]:
                        merged[key].append(v)
            else:
                # Otherwise, simply update the value in the merged dictionary
                merged[key] = value
        else:
            merged[key] = value
    return merged
