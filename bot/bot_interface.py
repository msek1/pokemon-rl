from dataclasses import dataclass
from typing import Optional, List
from poke_env.environment.battle import Battle

@dataclass
class PokemonMove:
    name: str

@dataclass
class PokemonAbility:
    name: str

@dataclass
class PokemonItem:
    name: Optional[str]
    holding: bool

@dataclass
class PokemonStats:
    attack: int
    defence: int
    specialAttack: int
    specialDefence: int
    speed: int
    hitPoints: int

@dataclass
class PokemonBoosts:
    attack: int
    defence: int
    specialAttack: int
    specialDefence: int
    speed: int
    accuracy: int
    evasiveness: int
    critical: int

@dataclass
class PokemonStatus:
    majorStatus: Optional[str] # Poison, Toxic, Paralysis, Sleep, Burn, Freeze
    statusTurns: int
    isConfused: bool
    isInfatuated: bool
    isTrapped: bool
    isDrowsy: bool
    isSubstituted: bool
    perishSongTurns: int
    isSeeded: bool
    disabledMove: Optional[int]
    isTaunted: bool
    isTormented: bool
    isRecharging: bool
    isCharging: bool
    didProtect: bool

@dataclass
class BattleTerrain:
    weather: Optional[str] # Sun, Rain, Sandstorm, Hail
    terrain: Optional[str] # Electric, Grassy, Psychic, Misty

    isTrickRoom: int
    isMagicRoom: int
    isWonderRoom: int
    isGravity: int

    teamSpikesLayers: int
    teamPoisonLayers: int
    teamStealthRocks: bool
    teamStickyWeb: bool
    oppSpikesLayers: int
    oppPoisonLayers: int
    oppStealthRocks: bool
    oppStickyWeb: bool

@dataclass
class Pokemon:
    name: str
    baseStats: Optional[PokemonStats]
    stats: Optional[PokemonStats]
    boosts: PokemonBoosts
    types: List[str]
    known_moves: List[PokemonMove]
    ability: Optional[PokemonAbility]
    item: Optional[PokemonItem]
    status: PokemonStatus

@dataclass
class NetworkBattle:
    ownPokemon: List[Pokemon]
    ownActive: Optional[Pokemon]
    oppPokemon: List[Pokemon]
    oppActive: Optional[Pokemon]
    field: BattleTerrain

# class EnvironmentInterface:
#     def __init__(self, pkmnEncoderNet: EncoderNetwork, ) -> None:
#         pass

class EnvironmentMapper:
    def mapBattle(self, battle: Battle) -> NetworkBattle:
        pass

    def mapSinglePokemon(self):
        pass