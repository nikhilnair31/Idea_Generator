import time
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate('keys/ideahub31-firebase-adminsdk-yl59k-f6da5b2634.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def firestore_clean():
    gen_ref = db.collection(u"generated")
    pos_ref = db.collection(u"posts")
    # query = gen_ref.order_by("utc") #.collection("posts").where("displayName", "==", "GPT2-Bot")
    query = pos_ref.where("displayName", "==", "GPT3-Bot")
    results = query.get()
    for obj in results:
        obj.reference.update({'uid': '5aCGwn68JWUpOv3QSwduabVqqG62'})

if __name__ == '__main__':
    firestore_clean()