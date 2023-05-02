from firebase_admin import firestore

import os
import time
import json
import traceback
from tqdm.auto import tqdm

from ..wikipedia_utils import get_extracted_html_with_page_title
from ..html_to_markdown_convertor import convert_html_to_markdown

from .db import (
    db,
    page_titles_to_download_collection,
    page_titles_in_progress_collection,
    page_titles_downloaded_collection,
    pages_collection,
)

MAX_RETRIES = 50


def page_downloader(batch_size=50):
    @firestore.transactional  # type: ignore
    def get_next_batch_of_page_titles(
            transaction, collection, in_progress_collection):
        title_docs = collection.limit(batch_size).get(transaction=transaction)
        if not title_docs:
            return

        return_items = []
        for title_doc in title_docs:
            if not title_doc:
                continue

            base64_page_title = title_doc.id
            page_title = title_doc.get('title')

            transaction.set(
                in_progress_collection.document(base64_page_title),
                {'title': page_title, 'started_at': time.time()}
            )
            transaction.delete(collection.document(base64_page_title))

            return_items.append((page_title, base64_page_title))

        return return_items

    progress_bar_items = 100_000
    print(
        f"The progress bar is fake, we don't know how many items are left, so we just use the number {progress_bar_items} to show a progress bar.")
    progress = tqdm(total=progress_bar_items)

    while True:
        items = None
        get_pate_title_retries = 0
        while True:
            try:
                transaction = db.transaction()
                items = get_next_batch_of_page_titles(
                    transaction,
                    page_titles_to_download_collection, page_titles_in_progress_collection
                )
                break
            except Exception as e:
                if get_pate_title_retries > MAX_RETRIES:
                    raise e
                print(f"--- Error: Retrying due to {e}")
                print("--- EndOfError")
                time.sleep(2)
                get_pate_title_retries += 1
        get_pate_title_retries = 0

        if not items:
            break
        if len(items) == 0:
            break

        batch = db.batch()
        for item in items:
            page_title, base64_page_title = item
            try:
                print(page_title)

                data = get_extracted_html_with_page_title(
                    page_title, additional_info=True)

                batch.set(
                    pages_collection.document(base64_page_title),
                    {
                        'base64_title': base64_page_title,
                        'markdown': convert_html_to_markdown(data['html']),
                        **data,
                        'title': page_title,
                    },
                    merge=True)
                batch.set(
                    page_titles_downloaded_collection.document(
                        base64_page_title),
                    {
                        'title': page_title,
                        'timestamp': time.time(),
                    })
                batch.delete(
                    page_titles_in_progress_collection.document(base64_page_title))

            except Exception as e:
                print(f"---Error on {item}:", str(e))
                error_traceback = traceback.format_tb(e.__traceback__)
                page_titles_in_progress_collection.document(base64_page_title).set({
                    'error': str(e),
                    'error_traceback': error_traceback,
                    'errored_at': time.time(),
                }, merge=True)
                print(error_traceback)
                print("---EndOfError")

            progress.update(1)
        batch.commit()

    print("Nothing left to download.")


def get_downloaded_page_markdown(base64_title):
    if not base64_title:
        raise ValueError("base64_title must be a non-empty string.")
    doc = pages_collection.document(base64_title).get()
    if not doc:
        return None
    return doc.get('markdown')
