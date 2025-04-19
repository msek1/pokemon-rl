from dataclasses import dataclass
from typing import Optional, List
import torch

STATUSES = ["poison", "toxic", "paralysis", "sleep", "burn", "freeze", "faint"]
WEATHER = ["sun", "rain", "snow", "sandstorm"]
TERRAIN = ["electric", "grassy", "psychic", "misty"]

@dataclass
class PokemonMove:
    name: str
    pp_left: int

@dataclass
class PokemonAbility:
    name: str

@dataclass
class PokemonItem:
    name: Optional[str] = None
    holding: bool = False

@dataclass
class PokemonStats:
    attack: int
    defence: int
    specialAttack: int
    specialDefence: int
    speed: int
    hitPoints: int

    def as_list(self):
        l = [self.hitPoints, self.attack, self.defence, self.specialAttack,
             self.specialDefence, self.speed]
        return [sum(l)] + l

@dataclass
class PokemonBoosts:
    attack: int = 0
    defence: int = 0
    specialAttack: int = 0
    specialDefence: int = 0
    speed: int = 0
    accuracy: int = 0
    evasiveness: int = 0
    critical: int = 0

    def to_vector(self):
        return torch.Tensor([
            self.attack, self.defence, self.specialAttack, self.specialDefence,
            self.speed, self.accuracy, self.evasiveness, self.critical
        ])

@dataclass
class PokemonStatus:
    majorStatus: Optional[str] = None # Poison, Toxic, Paralysis, Sleep, Burn, Freeze
    statusTurns: int = 0

    def to_vector(self):
        result = torch.zeros(8)
        if self.majorStatus is not None:
            result[STATUSES.index(self.majorStatus)] = 1
            result[7] = float(self.statusTurns)
        return result

@dataclass
class PokemonEphemeralStatus:
    '''Status can be removed by switching out and only applies to the active pokemon'''
    isConfused: bool = False
    isInfatuated: bool = False
    isTrapped: bool = False
    isDrowsy: bool = False
    isSubstituted: bool = False
    perishSongTurns: int = 0
    isSeeded: bool = False
    disabledMove: Optional[int] = None
    isTaunted: bool = False
    isTormented: bool = False
    isRecharging: bool = False
    isCharging: bool = False
    didProtect: bool = False
    isFirstTurn: bool = False

    def to_vector(self):
        result = torch.zeros(15)
        if self.disabledMove is not None:
            result[0] = 1
            result[1] = self.disabledMove
        result[2:] = torch.Tensor(
            [self.isConfused, self.isInfatuated, self.isTrapped, self.isDrowsy, self.isSubstituted, self.perishSongTurns,
             self.isSeeded, self.isTaunted, self.isTormented, self.isRecharging, self.isCharging, self.didProtect, self.isFirstTurn]
        )
        return result

@dataclass
class BattleTerrain:
    weather: Optional[str] = None # Sun, Rain, Sandstorm, Hail
    weatherTurns: int = 0
    terrain: Optional[str] = None # Electric, Grassy, Psychic, Misty
    terrainTurns: int = 0

    isTrickRoom: int = 0
    isMagicRoom: int = 0
    isWonderRoom: int = 0
    isGravity: int = 0

    teamSpikesLayers: int = 0
    teamPoisonLayers: int = 0
    teamStealthRocks: bool = False
    teamStickyWeb: bool = False
    teamTailwind: int = 0
    teamSafeguard: int = 0
    teamAuroraVeil: int = 0
    teamReflect: int = 0
    teamLightScreen: int = 0
    oppSpikesLayers: int = 0
    oppPoisonLayers: int = 0
    oppStealthRocks: bool = False
    oppStickyWeb: bool = False
    oppTailwind: int = 0
    oppSafeguard: int = 0
    oppAuroraVeil: int = 0
    oppReflect: int = 0
    oppLightScreen: int = 0

    def to_vector(self):
        result = torch.zeros(32)
        if self.weather is not None:
            result[WEATHER.index(self.weather)] = 1
            result[4] = self.weatherTurns
        elif self.terrain is not None:
            result[5 + TERRAIN.index(self.terrain)] = 1
            result[9] = self.terrainTurns
        result[10:] = torch.Tensor(
            [
             self.isTrickRoom, self.isMagicRoom, self.isWonderRoom, self.isGravity,

             self.teamSpikesLayers, self.teamPoisonLayers, self.teamStealthRocks, self.teamStickyWeb,
             self.teamTailwind, self.teamSafeguard, self.teamAuroraVeil, self.teamReflect, self.teamLightScreen,

             self.oppSpikesLayers, self.oppPoisonLayers, self.oppStealthRocks, self.oppStickyWeb,
             self.oppTailwind, self.oppSafeguard, self.oppAuroraVeil, self.oppReflect, self.oppLightScreen
            ]
        )
        return result

@dataclass
class Pokemon:
    name: str
    baseStats: PokemonStats
    stats: Optional[PokemonStats]
    boosts: PokemonBoosts
    types: List[str]
    known_moves: List[PokemonMove]
    ability: Optional[PokemonAbility]
    item: Optional[PokemonItem]
    status: PokemonStatus
    current_hp: float

@dataclass
class NetworkBattle:
    ownPokemon: List[Pokemon]
    ownActive: Optional[Pokemon]
    ownActiveStatus: Optional[PokemonEphemeralStatus]
    oppPokemon: List[Pokemon]
    oppActive: Optional[Pokemon]
    oppActiveStatus: Optional[PokemonEphemeralStatus]
    field: BattleTerrain
