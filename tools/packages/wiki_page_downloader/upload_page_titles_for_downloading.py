from .db import (
    db,
    page_titles_to_download_collection,
)

import os
import time
from tqdm.auto import tqdm
import base64
import random


def upload_page_titles_for_downloading(limit=None):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    titles_file_path = os.path.join(file_dir, 'all-titles-in-ns0.txt')
    if not os.path.isfile(titles_file_path):
        raise Exception(f"Please place the all titles file at {titles_file_path}. You can get this file from https://dumps.wikimedia.org/backup-index.html. Select on a dump, and find the file under 'List of page titles in main namespace'.")

    print(f"Found titles file {titles_file_path}.")
    started_at = time.time()
    print(f"Loading all titles from {titles_file_path}...")
    with open(titles_file_path, "r") as file:
        titles = [line.strip() for line in file.readlines()]
    print("Titles count:", len(titles))
    print(f"Titles loaded. ({time.time() - started_at:.4f}s)")

    if limit:
        print(f"Limiting to {limit} titles.")
        # titles = titles[:limit]
        titles = random.sample(titles, limit)

    print(f"Uploading page titles to Firestore...")
    started_at = time.time()
    titles_collection = page_titles_to_download_collection

    batch_size = 500
    batch = db.batch()
    for i, title in tqdm(enumerate(titles), total=len(titles)):
        base64_title = base64.urlsafe_b64encode(title.encode('utf-8')).decode()
        doc_ref = titles_collection.document(base64_title)
        batch.set(doc_ref, {'title': title, 'added_at': time.time()}, merge=True)
        if (i + 1) % batch_size == 0:
            batch.commit()
            batch = db.batch()
    batch.commit()

    print(f"Done. ({time.time() - started_at:.4f}s)")

    # Normally this will just timeout.
    # print("Page titles in Firestore:", len(titles_collection.get()))
