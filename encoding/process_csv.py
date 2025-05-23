import pandas as pd # type: ignore
import numpy as np

def scale(series):
    return np.log1p(series)

def scale_min_max(series):
    series = series.astype(str).str.replace('%', '', regex=True)  
    series = pd.to_numeric(series, errors="coerce")
    min_val = series.min(skipna=True)
    max_val = series.max(skipna=True)

    return (series - min_val) / (max_val - min_val)


def augment_HP(df_raw, n = 20):
    rows = []
    for _, row in df_raw.iterrows():
        max_hp = row["HP"]
        for _ in range(n):
            new_row = row.copy()
            new_row["HP"] = np.random.uniform(0, max_hp)
            rows.append(new_row)
    
    return pd.DataFrame(rows)

def get_input(data, df_raw, one_hot_keys):
    data_keys = data.keys()

    for _, row in df_raw.iterrows():
        for k in row.index:
            if (k in data_keys):
                data[k].append(row[k])

    df = pd.DataFrame(data)
    scaled_keys = []
    for k in data_keys:
        if (k not in one_hot_keys):
            scaled_key = k + "_scaled"
            scaled_keys.append(scaled_key)
            df[scaled_key] = scale_min_max(df[k])

    # generate one hot columns for each type in one_hot_keys
    df_complete = pd.get_dummies(df, columns=one_hot_keys)

    # extract only the one_hot_cols and the scaled_keys columns
    one_hot_cols = []
    for col in df_complete.columns:
        for start in one_hot_keys:
            if (col.startswith(start)):
                one_hot_cols.append(col)

    return df_complete[one_hot_cols + scaled_keys] 

df1 = pd.read_csv("data/moves.csv")
data = {
    "type": [],
    "category": [],
    "power": [],
    "accuracy": [],
    "pp": [],
    # "z-effect": [],
    "priority": [],
    "crit": []
}
moves_df = get_input(data, df1, ["type", "category"])

df1 = pd.read_csv("data/pokedex.csv")
df2 = pd.read_csv("data/pokemon.csv")
df_joined = df1.merge(df2, left_on="Name", right_on="name", how="inner")
pokemon_df = augment_HP(df_joined)
    
data = {
    "Type 1": [],
    "Type 2": [],
    "Total": [],
    "HP": [],
    "Attack": [],
    "Defense": [],
    "Sp. Attack": [],
    "Sp. Defense": [],
    "Speed": [],
    "against_bug": [],
    "against_dark": [],
    "against_dragon": [],
    "against_electric": [],
    "against_fairy": [],
    "against_fight": [],
    "against_fire": [],
    "against_flying": [],
    "against_ghost": [],
    "against_grass": [],
    "against_ground": [],
    "against_ice": [],
    "against_normal": [],
    "against_poison": [],
    "against_psychic": [],
    "against_rock": [],
    "against_steel": [],
    "against_water": []
}


pokemon_df = get_input(data, df_joined, ["Type 1", "Type 2"])




    