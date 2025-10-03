#TO RUN:
# python live_client_eeg.py  --host 127.0.0.1 --port 5000 --fs 256


import argparse
import time
import sys
import pandas as pd
import numpy    as np
import requests
import os
import threading

from datetime import datetime, timedelta, timezone
from typing   import List, Tuple, Any, Dict

import firebase_admin
from firebase_admin import credentials, firestore
from dotenv         import load_dotenv
load_dotenv()

sys.path.insert(0, "./src")
print("path: ", sys.path)
#to make it find awear_neuroscience folder
from awear_neuroscience.data_extraction.firestore_loader import query_eeg_data, process_eeg_records

cred = credentials.Certificate(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
firebase_admin.initialize_app(cred)
firestore_client = firestore.Client()
print("FIREBASE INITIALIZED")

#--------- each device will be linked to an email ----------
emails_str = os.getenv("EMAILS", "")
if not emails_str:
    print("No EMAILS found in environment variables. Using DOCUMENT_NAME as fallback.")
    available_emails = [os.getenv("DOCUMENT_NAME")]
else:
    available_emails = [email.strip() for email in emails_str.split(",")]
    if len(available_emails) > 1:
        print("more than your own email, smt wong")
    
print("available emails ", available_emails)
email = available_emails[0]
print("only email: ", email)

#------------ NEW: for live data collection ---------------
def on_snapshot(col_snap, changes, read_time):
    print("XXXX CALLBACK")
    raw_record: List[Dict[str, Any]] = []
    for change in changes:
        print("XXXXXXXX CHANGE : \n", change)
        print("CHANGE TYPE NAME: ", change.type.name)
        if change.type.name in ("ADDED"):
            doc  = change.document
            raw_data = doc.to_dict()
            raw_record.append(raw_data) 
            print(f"[{read_time}] New/updated: {doc.id} -> data: {list(raw_data.keys())}")
            long_df     = process_eeg_records(raw_record, return_long=True)
            raw_record.pop()
            ordered     = long_df.sort_values(["segment", "time_sample"], kind="mergesort")
            samples = ordered["waveform_value"].astype(np.float32).to_numpy()
            
            # drop NaNs (replace with 0.0) (AVOID THIS?)
            if np.isnan(samples).any():
                print("[WARN] NaNs in waveform_value; filling with 0.0")
                samples = np.nan_to_num(samples, nan=0.0)
            #Samples_all is already a "small" chunk
            try:
                r = requests.post(GLOBAL_URL, json={"samples": samples.tolist()}, timeout=2.0)
                if r.status_code != 200:
                    print(f"[WARN] Bad status {r.status_code}: {r.text[:200]}...")
            except Exception as e:
                print(f"[ERROR] POST failed: {e}")

            #print("DOC: ", doc)

def start_live_data_thread(): # only get new data
    start_time = datetime.now(timezone.utc) - timedelta(seconds=5)

    collection_name = os.getenv("COLLECTION_NAME")
    document_name   = os.getenv("DOCUMENT_NAME")
    sub_coll_name   = os.getenv("SUB_COLLECTION_NAME")
    
    col_ref          = firestore_client.collection(collection_name)
    collection_query = col_ref.document(document_name).collection(sub_coll_name) #selects live data???
  
    listener = collection_query.on_snapshot(on_snapshot)

    while True:
        print("INSIDE TRUE LOOP")
        print('', end='', flush=True)
        time.sleep(1)


    #threading.Event().wait()

GLOBAL_URL = ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",         type=str,   default="127.0.0.1", help="Flask server host")
    ap.add_argument("--port",         type=int,   default=5000,        help="Flask server port")
    ap.add_argument("--fs",           type=float, default=256.0,       help="Sample rate (Hz)")
    args = ap.parse_args()

    global GLOBAL_URL
    GLOBAL_URL = f"http://{args.host}:{args.port}/ingest"
    print(f"Streaming to {GLOBAL_URL} at fs={args.fs} Hz, (Ctrl-C to stop)")

    #------------- START LIVE DATA THREAD ----------------
    start_live_data_thread()
    

if __name__ == "__main__":
    main()
