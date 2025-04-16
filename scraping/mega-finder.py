# Gets the megas and zmoves from the games and saves to megas.txt and zmoves.txt

import os

DIRS = ["nu", "ou", "ru", "ubers", "uu"]
PREF = "scraping/games/"

def list_files():
    for dir in DIRS:
        for file in os.listdir(PREF+dir):
            yield f"{PREF+dir}/{file}"

mega_games = 0
z_games = 0
either_one = set()

def find_megas(filename, output):
    global mega_games
    with open(filename) as f:
        lines = f.readlines()
    saw = False
    for l in lines:
        if "-mega" in l.lower():
            if not saw:
                saw = True
                output.write(filename + "::\n")
            output.write(l)
    if saw:
        mega_games += 1
        either_one.add(filename)
        

def find_Z(filename, output):
    global z_games
    with open(filename) as f:
        lines = f.readlines()
    saw = False
    for l in lines:
        if "-z" in l.lower():
            if not saw:
                saw = True
                output.write(filename + "::\n")
            output.write(l)
    if saw:
        z_games += 1
        either_one.add(filename)

with open("megas.txt", "w") as f:
    for file in list_files():
        find_megas(file, f)

with open("zmoves.txt", "w") as f:
    for file in list_files():
        find_Z(file, f)

print(mega_games, z_games)
print(len(either_one))
