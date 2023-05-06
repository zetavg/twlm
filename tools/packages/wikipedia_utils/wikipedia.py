import requests
import time
import json

from ..html_to_markdown_convertor import convert_html_to_markdown


def wiki_request(params, lang='zh-tw', debug=False):
    if lang.startswith('zh-'):
        api_url = 'https://zh.wikipedia.org/w/api.php'
    else:
        api_url = f"https://{lang}.wikipedia.org/w/api.php"

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


def get_extracted_html_with_page_title(page_title):
    params = {
        'titles': page_title,
        'redirects': True,
        'converttitles': True,
        'prop': ['extracts']
    }

    response_json = wiki_request(params)
    page_data = list(response_json['query']['pages'].values())[0]

    if 'missing' in page_data:
        raise Exception(f"Page '{page_title}' not found.")
    try:
        page = list(response_json['query']['pages'].values())[0]
        html = page['extract']
        return html
    except Exception as e:
        raise Exception(f"Error when getting page '{page_title}': {response_json} - {e}")


def get_page_markdown_with_page_title(page_title):
    html = get_extracted_html_with_page_title(page_title)
    return convert_html_to_markdown(html)


def get_page_data(page_title, lang='zh-tw'):
    params = {
        'titles': page_title,
        'redirects': True,
        'converttitles': True,
        'prop': ['extracts', 'info', 'coordinates'],
        'inprop': ['displaytitle', 'varianttitles'],
    }

    response_json = wiki_request(params, lang=lang)
    page_data = list(response_json['query']['pages'].values())[0]

    if 'missing' in page_data:
        raise Exception(f"Page '{page_title}' not found.")
    try:
        page = list(response_json['query']['pages'].values())[0]
        html = page['extract']
        markdown = convert_html_to_markdown(html)

        original_title = page['title']

        title = page['title']

        if 'varianttitles' in page and lang in page['varianttitles']:
            title = page['varianttitles'][lang]

        coordinates = page.get('coordinates')
        if coordinates and len(coordinates) > 0:
            coordinate = {
                'lat': coordinates[0]['lat'],
                'lon': coordinates[0]['lon'],
                'globe': coordinates[0]['globe'],
            }
        else:
            coordinate = None

        return {
            'title': title,
            'pageid': page.get('pageid'),
            'html': html,
            'markdown': markdown,
            'coordinate': coordinate,
            'length': page.get('length'),
            'touched': page.get('touched'),
            'lastrevid': page.get('lastrevid'),
            'original_title': original_title,
            # 'varianttitles': page.get('varianttitles'),
        }
    except Exception as e:
        raise Exception(f"Error when getting page '{page_title}': {response_json} - {e}")
