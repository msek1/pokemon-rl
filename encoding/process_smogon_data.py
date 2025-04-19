import json

with open("data/smogon_data.json") as f:
    smogon = json.load(f)

SM_GEN = "SM"
STANDARD = "Standard"

abilities_list = []
item_list = []

def normalize_name(name):
    name = name.replace(" ", "")
    name = name.replace("-", "")
    return name.lower()

for ab in smogon["abilities"]:
    if ab["isNonstandard"] == STANDARD and SM_GEN in ab["genfamily"]:
        abilities_list.append(normalize_name(ab["name"]))

for item in smogon["items"]:
    if item["isNonstandard"] == STANDARD and SM_GEN in item["genfamily"]:
        item_list.append(normalize_name(item["name"]))

with open("data/abilities.txt", "w") as f:
    for ab in abilities_list:
        f.write(ab)
        f.write('\n')

with open("data/items.txt", "w") as f:
    for item in item_list:
        f.write(item)
        f.write('\n')
