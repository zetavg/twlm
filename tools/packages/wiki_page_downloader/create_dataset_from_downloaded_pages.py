import os
import huggingface_hub
from datasets import Dataset
from firebase_admin import firestore
from tqdm import tqdm

from .db import firestore

file_dir = os.path.dirname(os.path.abspath(__file__))


def create_dataset_from_downloaded_pages(dataset_name):
    os.makedirs(os.path.join(file_dir, 'datasets'), exist_ok=True)
    api = huggingface_hub.HfApi()
    user_info = api.whoami()
    print(f"Current login to HF as: {user_info['name']}.")

    def get_data():
        # Need this to be here rather then import it form .db.
        # Otherwise will get this error:
        # "_pickle.PicklingError: Pickling client objects is explicitly not supported."
        db = firestore.client()
        pages_collection = db.collection("pages")

        pages_count = pages_collection.count().get()[0][0].value
        collection = pages_collection.order_by(
            "title",
            direction=firestore.Query.ASCENDING
        )
        batch_size = 1000
        batches_count = int(pages_count / batch_size)

        progress = tqdm(total=batches_count)
        last_doc = None
        more_docs = True

        while more_docs:
            if last_doc:
                query = collection.start_after(last_doc).limit(batch_size)
            else:
                query = collection.limit(batch_size)

            docs = query.stream()
            have_docs = False
            for doc in docs:
                last_doc = doc
                if not have_docs:
                    tqdm.write(doc.get('title'))
                    have_docs = True
                doc = doc.to_dict()
                yield {
                    # 'title': doc.get('title'),
                    'pageid': doc.get('pageid'),
                    'html': doc.get('html'),
                    'markdown': doc.get('markdown'),
                    'coordinate': doc.get('coordinate') or doc.get('coordinates'),
                    'length': doc.get('length'),
                    'touched': doc.get('touched'),
                    'lastrevid': doc.get('lastrevid'),
                    'original_title': doc.get('original_title') or doc.get('title'),
                }

            progress.update(1)

            if not have_docs:
                more_docs = False

    print(next(get_data()))
    ds = Dataset.from_generator(get_data)
    print("Saving dataset...")
    dataset_save_to_path = os.path.join(file_dir, 'datasets', dataset_name)
    ds.save_to_disk(dataset_save_to_path)
    print(f"Dataset saved to: {dataset_save_to_path}. Pages count: {len(ds)}.")
    print("Pushing dataset to the hub...")
    ds.push_to_hub(dataset_name, private=True)
    print(f"Done. Pages count: {len(ds)}.")
