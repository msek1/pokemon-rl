import requests
import pandas as pd
import json
import datetime
import time
import random
from numpy.random import choice
import os

URL_BASE = "https://replay.pokemonshowdown.com/search.json?format={0}&before={1}"
game_format_base = "gen7{0}"
formats = ["ou", "uu", "ru", "nu", "ubers"]
DESIRED_COLUMNS = ["id", "format", "rating", "uploadtime"]
FILENAME = "game_samples.csv"
DESIRED_SAMPLES = 16500

def get_page(format: str, timestamp: int = None) -> pd.DataFrame:
    if timestamp is None:
        timestamp = datetime.datetime.now().timestamp()
    fmt_str = game_format_base.format(format)
    url = URL_BASE.format(fmt_str, timestamp)
    resp = requests.get(url)
    return pd.json_normalize(json.loads(resp.text))

def dataframe_mask(df):
    return df["rating"].notna() & (df["rating"] > 1000) & df["password"].isna() & (df["private"] == 0)

def dataframe_relevant_columns(df):
    return df[DESIRED_COLUMNS]

time_range_min = int(datetime.datetime(2017,6,1).timestamp())
time_range_max = int(datetime.datetime(2021,1,1).timestamp())

acc_df = None
if os.path.exists(FILENAME):
    acc_df = pd.read_csv(FILENAME)

samples = 5
sample_iter = 0

while acc_df is None or acc_df.shape[0] < DESIRED_SAMPLES:
    for i in range(samples):
        ts = random.randint(time_range_min, time_range_max)
        format = choice(formats, 1, p=[0.25,0.3,0.15,0.1,0.2])[0]
        print(f"{sample_iter+1},{i+1}: Sampling {format} at {ts}")
        
        p = get_page(format, ts)
        mask = dataframe_mask(p)

        if acc_df is None:
            acc_df = dataframe_relevant_columns(p[mask])
        else:
            acc_df = pd.merge(acc_df, dataframe_relevant_columns(p[mask]), how="outer", on=DESIRED_COLUMNS)
        
        if i < samples - 1:
            time.sleep(random.uniform(2,7))
    sample_iter += 1

    print(acc_df.shape)
    acc_df.to_csv(FILENAME, index=False)