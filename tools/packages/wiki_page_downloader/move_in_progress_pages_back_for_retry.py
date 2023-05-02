from firebase_admin import firestore
import time

from .db import (
    db,
    page_titles_to_download_collection,
    page_titles_in_progress_collection,
)

MAX_RETRIES = 50
BATCH_SIZE = 20


def move_in_progress_pages_back_for_retry():
    moved_count = 0

    @firestore.transactional  # type: ignore
    def move_in_progress_page_back_to_pending(
            transaction, collection, in_progress_collection):
        nonlocal moved_count
        title_docs = in_progress_collection.limit(
            BATCH_SIZE).get(transaction=transaction)
        if not title_docs:
            return False

        for title_doc in title_docs:
            if not title_doc:
                continue
            base64_page_title = title_doc.id
            page_title = title_doc.get('title')

            doc_ref = collection.document(base64_page_title)
            transaction.set(doc_ref, {
                'title': page_title,
                'added_at': time.time(),
                'retry': True
            })

            transaction.delete(
                in_progress_collection.document(base64_page_title))

            moved_count += 1

        return True

    while True:
        transaction = db.transaction()
        should_continue = move_in_progress_page_back_to_pending(
            transaction,
            page_titles_to_download_collection, page_titles_in_progress_collection
        )
        if not should_continue:
            break

    print(f"Moved {moved_count} pages back for retry.")
