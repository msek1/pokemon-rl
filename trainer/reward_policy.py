from dataclasses import dataclass

@dataclass
class RewardPolicy:
    damage_proportion: float = 1
    own_damage_factor: float = 0.8
    healing_proportion: float = 1
    faint_reward: float = 2
    win_reward: float = 15
