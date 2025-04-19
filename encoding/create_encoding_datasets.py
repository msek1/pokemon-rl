import pandas as pd # type: ignore
import numpy as np
import json
from dataclasses import dataclass
from typing import List
import torch
from torch import Tensor
import pickle

TYPE_ORDER = [
    "Normal","Fire","Water","Grass","Fighting","Flying",
    "Rock","Ground","Steel","Poison","Electric","Bug",
    "Ice","Dark","Ghost","Psychic","Dragon","Fairy"]
MOVE_CATEGORIES = ["Physical", "Special", "Status"]

CURRENT_HP_INDEX = 7
CURRENT_PP_INDEX = 8
MIN_EFFECTIVENESS = 2**-10

effectiveness_data = {}

def calc_effectiveness_vector(atk_effectiveness):
    atk_effectiveness = sorted(atk_effectiveness, key=lambda x: TYPE_ORDER.index(x[0]))
    return np.array([max(x[1], MIN_EFFECTIVENESS) for x in atk_effectiveness])

with open("data/smogon_data.json") as f:
    smogon_data = json.load(f)
type_data = smogon_data["types"]

for t in type_data:
    effectiveness_data[t["name"]] = calc_effectiveness_vector(t["atk_effectives"])

def scale(value):
    return np.log10(value + 1)

def get_log_effectiveness_vector(type):
    return  np.log2(effectiveness_data[type])

def normalize_name(name):
    name = name.replace(" ", "")
    name = name.replace("-", "")
    name = name.replace(",", "")
    name = name.replace("'", "")
    return name.lower()

@dataclass
class PokemonEncodingData:
    types: List[str]
    base_stat_total: int
    base_hp: int
    base_atk: int
    base_def: int
    base_spatk: int
    base_spdef: int
    base_speed: int
    current_hp_percent: float = 1

    def to_vector(self):
        result = np.zeros(8 + 18 + 18) # 7 stats + current HP, 18 for pkm type, 18 for effectiveness
        stats = np.array([
            self.base_stat_total, self.base_hp, self.base_atk, self.base_def,
            self.base_spatk, self.base_spdef, self.base_speed]
        )
        types = np.zeros(18)
        log_effectiveness = np.zeros(18)
        for t in self.types:
            types[TYPE_ORDER.index(t)] = 1
            log_effectiveness = log_effectiveness + get_log_effectiveness_vector(t)
        
        result[:7] = np.log10(stats)
        result[7] = self.current_hp_percent
        result[8:8+18] = types
        result[8+18:] = log_effectiveness
        return Tensor(result)

@dataclass
class MoveEncodingData:
    power: int
    category: int # 0 = Physical, 1 = Special, 2 = Status
    accuracy: float
    pp: int
    priority: int
    crit: int
    effectiveness: Tensor

    def to_vector(self):
        result = torch.zeros(5 + 3 + 1 + 18) # Stats + Category + PP left + effectiveness
        result[:5] = Tensor(
            [np.log10(float(self.power) + 1), self.accuracy, np.log2(float(self.pp)), self.priority, self.crit]
        )
        result[5 + self.category] = 1
        result[8] = 1
        result[9:] = self.effectiveness
        return Tensor(result)

_moves_df = pd.read_csv("data/moves.csv")
_pokemon_df = pd.read_csv("data/pokedex.csv")

def augment_column_as_uniform(vector, column, N = 40):
    matrix = Tensor.repeat(vector.reshape(1, *vector.shape), (N,1))
    matrix[1:,column] = torch.rand(N-1)
    return Tensor(matrix)

def create_pokemon_dataset_tensor():
    vectors = []
    for _,row in _pokemon_df.iterrows():
        types = [row["Type 1"]]
        if not pd.isna(row["Type 2"]):
            types.append(row["Type 2"])
        d = PokemonEncodingData(
            types,
            *row[["Total","HP","Attack","Defense","Sp. Attack","Sp. Defense","Speed"]])
        vectors.append(augment_column_as_uniform(d.to_vector(), CURRENT_HP_INDEX))
    return torch.concat(vectors,axis=0)

def create_move_dataset_tensor():
    vectors = []
    for _,row in _moves_df.iterrows():
        effectiveness = get_log_effectiveness_vector(row["type"])
        if row["move"] == "Flying Press":
            effectiveness += get_log_effectiveness_vector("Flying") # Dual-type
        elif row["move"] == "Freeze-Dry":
            effectiveness[TYPE_ORDER.index("Water")] = 1 # 2x effective against water

        if pd.isna(row["category"]):
            continue

        power = row["power"] if row["power"] != "â€”" else 10000
        acc = float(row["accuracy"][:-1]) if row["accuracy"] != "â€”" else 1000

        move = MoveEncodingData(
            power, MOVE_CATEGORIES.index(row["category"]),
            acc / 100, row["pp"], row["priority"], row["crit"],
            Tensor(effectiveness)
        )
        
        vectors.append(augment_column_as_uniform(move.to_vector(), CURRENT_PP_INDEX, N=20))
        
    return torch.concat(vectors,axis=0)

def create_move_dict():
    d = {}
    for _,row in _moves_df.iterrows():
        effectiveness = get_log_effectiveness_vector(row["type"])
        if row["move"] == "Flying Press":
            effectiveness += get_log_effectiveness_vector("Flying") # Dual-type
        elif row["move"] == "Freeze-Dry":
            effectiveness[TYPE_ORDER.index("Water")] = 1 # 2x effective against water

        if pd.isna(row["category"]):
            continue

        power = row["power"] if row["power"] != "â€”" else 10000
        acc = float(row["accuracy"][:-1]) if row["accuracy"] != "â€”" else 1000

        move = MoveEncodingData(
            power, MOVE_CATEGORIES.index(row["category"]),
            acc / 100, row["pp"], row["priority"], row["crit"],
            Tensor(effectiveness)
        )
        d[normalize_name(row["move"])] = move
    with open("bot_data/move_data.pkl", "wb") as f:
        pickle.dump(d, f)
