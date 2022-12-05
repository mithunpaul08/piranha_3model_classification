import pandas as pd

def fn_convert():
    df = pd.read_json('output/per_signature_retrieved_emails.json')
    df.to_json("output/per_signature_retrieved_emails.jsonl", orient="records", lines=True)

fn_convert()