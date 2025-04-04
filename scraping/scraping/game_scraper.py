import pandas as pd
import os
import requests
import time
import random

BASE_URL = "https://replay.pokemonshowdown.com/{0}.log"

games = pd.read_csv("game_samples.csv")
for i, game_id in enumerate(games["id"]):
    if os.path.exists(f"games/{game_id}.log"):
        continue
    print(f"{i}: Requesting Game ID {game_id}")
    url = BASE_URL.format(game_id)
    resp = requests.get(url)
    with open(f"games/{game_id}.log", "w") as f:
        f.write(resp.text)
    time.sleep(random.uniform(0.5,2))
