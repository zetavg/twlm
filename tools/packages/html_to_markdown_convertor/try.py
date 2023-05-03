import os
from html_to_markdown_convertor import convert_html_to_markdown

file_dir = os.path.dirname(os.path.abspath(__file__))
sample_file_path = os.path.join(file_dir, 'try.html')

with open(sample_file_path, 'r') as f:
    sample_html = f.read()
    markdown = convert_html_to_markdown(sample_html)
    print(markdown)
