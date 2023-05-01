import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from tqdm.auto import tqdm
import os
import traceback
import time
import json
import wikipedia
wikipedia.set_lang('zh-tw')

file_dir = os.path.dirname(os.path.abspath(__file__))
credentials_key_path = os.path.join(file_dir, 'serviceAccountKey.json')

cred = credentials.Certificate(credentials_key_path)
firebase_admin.initialize_app(cred)

db = firestore.client()


MAX_RETRIES = 50
BATCH_SIZE = 20


@firestore.transactional
def get_and_delete_next_page_title_in_transaction(
        transaction, collection, in_progress_collection):
    title_docs = collection.limit(BATCH_SIZE).get(transaction=transaction)
    if not title_docs:
        return

    return_items = []
    for title_doc in title_docs:
        if not title_doc:
            continue
        base64_page_title = title_doc.id
        page_title = title_doc.get('title')

        doc_ref = in_progress_collection.document(base64_page_title)
        transaction.set(doc_ref, {'title': page_title})

        transaction.delete(collection.document(base64_page_title))

        return_items.append((page_title, base64_page_title))
    return return_items


progress_bar_items = 100_000
print(
    f"The progress bar is fake, we don't know how many items are left, so we just use the number {progress_bar_items} to show a progress bar.")
progress = tqdm(total=100_000)
page_titles_to_download_collection = db.collection('page_titles_to_download')
page_titles_in_progress_collection = db.collection("page_titles_in_progress")
page_titles_saved_collection = db.collection("page_titles_saved")
pages_collection = db.collection("pages")

while True:
    items = None
    retry = 0
    while True:
        try:
            transaction = db.transaction()
            items = get_and_delete_next_page_title_in_transaction(
                transaction,
                page_titles_to_download_collection, page_titles_in_progress_collection
            )
            break
        except Exception as e:
            if retry > MAX_RETRIES:
                raise e
            print(f"---Error: Retrying due to {e}")
            print("---EndOfError")
            time.sleep(2)
            retry += 1

    if not items:
        break
    if len(items) == 0:
        break

    batch = db.batch()
    for item in items:
        try:
            page_title, base64_page_title = item
            # print(page_title)

            wikipedia_page = wikipedia.page(page_title)
            page_data = {
                'title': page_title,
                'summary': wikipedia_page.summary,
                'sections': {title: wikipedia_page.section(title) for title in wikipedia_page.sections},
                'content': wikipedia_page.content,
                # 'html': wikipedia_page.html(),
                # 'images': [],
                # 'categories': [],
                # 'links': [],
                # 'references': [],
                # 'coordinates': [],
                'revision_id': wikipedia_page.revision_id,
            }

            # Too many
            # try:
            #     images = wikipedia_page.images
            #     if images:
            #         page_data['images'] = images
            # except:
            #     pass

            # try:
            #     categories = wikipedia_page.categories
            #     if categories:
            #         page_data['categories'] = categories
            # except:
            #     pass

            # Too many
            # try:
            #     links = wikipedia_page.links
            #     if links:
            #         page_data['links'] = links
            # except:
            #     pass

            # try:
            #     references = wikipedia_page.references
            #     if references:
            #         page_data['references'] = references
            # except:
            #     pass

            # try:
            #     a, b = wikipedia_page.coordinates
            #     page_data['coordinates'] = [float(a), float(b)]
            # except:
            #     pass

            batch.set(
                pages_collection.document(base64_page_title),
                {
                    'base64_title': base64_page_title,
                    'title': page_title,
                    'data': json.dumps(page_data, ensure_ascii=False)
                },
                merge=True)
            batch.set(
                page_titles_saved_collection.document(base64_page_title),
                {
                    'title': page_title,
                })
            batch.delete(page_titles_in_progress_collection.document(
                base64_page_title))

        except Exception as e:
            print(f"---Error on {item}:", str(e))
            tb = traceback.format_tb(e.__traceback__)
            page_titles_in_progress_collection.document(base64_page_title).set({
                'error': str(e),
                'error_traceback': tb,
            }, merge=True)
            print(tb)
            print("---EndOfError")

        progress.update(1)
    batch.commit()
