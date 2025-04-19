import torch
import matplotlib.pyplot as plt
import pickle

def gen_point(N):
    x = torch.normal(torch.zeros(N))
    return x / torch.linalg.vector_norm(x)

ability_dim = 10
item_dim = 10

ability_encoding = {}
item_encoding = {}
constant_encodings = {"ability": ability_encoding, "item": item_encoding}

ability_encoding["UNKNOWN"] = torch.zeros(ability_dim)
item_encoding["UNKNOWN"] = torch.zeros(item_dim)
item_encoding["NONE"] = gen_point(item_dim)

with open("data/abilities.txt") as f:
    abilities = map(lambda s: s.strip(), f.readlines())
    for ab in abilities:
        ability_encoding[ab] = gen_point(ability_dim)

with open("data/items.txt") as f:
    items = map(lambda s: s.strip(), f.readlines())
    for item in items:
        item_encoding[item] = gen_point(item_dim)

with open("bot_data/constant_encodings.pkl", "wb") as f:
    pickle.dump(constant_encodings, f)

with open("bot_data/constant_encodings.pkl", "rb") as f:
    ce2 = pickle.load(f)

assert(torch.all(ce2["ability"]["UNKNOWN"] == ability_encoding["UNKNOWN"]))
assert(torch.all(ce2["ability"]["protean"] == ability_encoding["protean"]))
assert(torch.all(ce2["item"]["UNKNOWN"] == item_encoding["UNKNOWN"]))
assert(torch.all(ce2["item"]["NONE"] == item_encoding["NONE"]))
assert(torch.all(ce2["item"]["zoomlens"] == item_encoding["zoomlens"]))
