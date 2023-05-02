import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

file_dir = os.path.dirname(os.path.abspath(__file__))
credentials_key_path = os.path.join(file_dir, 'serviceAccountKey.json')

if not os.path.isfile(credentials_key_path):
    raise Exception(f"Please place your service account key at {credentials_key_path}. Get one from https://console.firebase.google.com/u/0/project/<your-project-name>/settings/serviceaccounts/adminsdk.")

cred = credentials.Certificate(credentials_key_path)
firebase_admin.initialize_app(cred)

db = firestore.client()

page_titles_to_download_collection = db.collection('page_titles_to_download')
page_titles_in_progress_collection = db.collection("page_titles_in_progress")
page_titles_downloaded_collection = db.collection("page_titles_downloaded")
pages_collection = db.collection("pages")
