import requests
import time
import json

from ..html_to_markdown_convertor import convert_html_to_markdown


def wiki_request(params, debug=False):
    api_url = 'https://zh.wikipedia.org/w/api.php'
    params['format'] = 'json'

    if 'action' not in params:
        params['action'] = 'query'

    if 'variant' not in params:
        params['variant'] = 'zh-tw'

    params = {
        k: "|".join(v) if isinstance(v, list) else v
        for k, v in params.items()
    }

    if debug:
        request_start = time.time()

    response = requests.get(
        api_url, params=params,  # headers=headers
    )

    if debug:
        print(f"({time.time() - request_start:.4f}) URL:", response.url)

    return response.json()


def get_extracted_html_with_page_title(page_title, additional_info=False):
    params = {
        'titles': page_title,
        'redirects': True,
        'converttitles': True,
        'prop': ['extracts']
    }

    if additional_info:
        params['prop'] += ['info', 'coordinates']

    response_json = wiki_request(params)
    page_data = list(response_json['query']['pages'].values())[0]

    if 'missing' in page_data:
        raise Exception(f"Page '{page_title}' not found.")
    try:
        page = list(response_json['query']['pages'].values())[0]
        html = page['extract']
        if not additional_info:
            return html

        coordinates = page.get('coordinates')
        if coordinates and len(coordinates) > 0:
            coordinates = {
                'lat': coordinates[0]['lat'],
                'lon': coordinates[0]['lon'],
                'globe': coordinates[0]['globe'],
            }
        else:
            coordinates = None

        return {
            'html': html,
            'coordinates': coordinates,
            'touched': page.get('touched'),
            'lastrevid': page.get('lastrevid'),
            'length': page.get('length'),
            'pageid': page.get('pageid'),
            'title': page.get('title'),
        }
    except Exception as e:
        raise Exception(f"Error when getting page '{page_title}': {response_json} - {e}")


def get_page_markdown_with_page_title(page_title):
    html = get_extracted_html_with_page_title(page_title)
    return convert_html_to_markdown(html)
