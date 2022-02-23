import os
import pandas as pd

files = os.listdir("cands/")
files = [os.path.join("cands", fpath) for fpath in files]

cand_ids = []
for fpath in files:
    try:
        tmp = pd.read_csv(fpath)
        cand_ids.extend(tmp["candidate_id"].to_list())
    except:
        print(f'Error loading data from {fpath}')

print(len(cand_ids))
df = pd.DataFrame(cand_ids, columns=['cand_id'])
print(df.head())
df.to_csv('cand_ids.csv')