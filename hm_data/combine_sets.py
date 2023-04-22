import argparse
import pandas as pd


path = "./hm_data"

def combine(seen, unseen):
    df_seen = pd.read_json(open(seen), lines=True)
    df_unseen = pd.read_json(open(unseen), lines=True)
    combined = pd.concat([df_seen, df_unseen])
    return combined.drop_duplicates(subset="img")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seen")
    parser.add_argument("-u", "--unseen")
    args = parser.parse_args()

    type = args.seen.split("_")[0]
    combined = combine(f"{path}/{args.seen}", f"{path}/{args.unseen}")
    combined.to_json(f"{path}/{type}_combined.jsonl", orient="records", lines=True)
    print(f"success combined {type}")