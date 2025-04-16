from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.teambuilder.teambuilder import Teambuilder
# from encoder_network import EncoderNetwork
import asyncio

# class MyBot(SimpleHeuristicsPlayer):

ou_team = '''Pelipper @ Life Orb  
Ability: Drizzle  
Tera Type: Water  
EVs: 248 HP / 252 SpA / 8 SpD  
Modest Nature  
IVs: 0 Atk  
- Hurricane  
- Roost  
- Ice Beam  
- Surf  

Ogerpon-Wellspring (F) @ Wellspring Mask  
Ability: Water Absorb  
Tera Type: Water  
EVs: 252 HP / 4 SpA / 252 Spe  
Timid Nature  
- Grassy Terrain  
- Giga Drain  
- Leech Seed  
- Spikes  

Azumarill @ Sitrus Berry  
Ability: Huge Power  
Tera Type: Water  
EVs: 252 HP / 252 Atk / 4 SpD  
Adamant Nature  
- Belly Drum  
- Aqua Jet  
- Play Rough  
- Superpower  

Darkrai @ Expert Belt  
Ability: Bad Dreams  
Tera Type: Dark  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Will-O-Wisp  
- Dark Pulse  
- Foul Play  
- Psyshock  

Alomomola @ Leftovers  
Ability: Regenerator  
Tera Type: Water  
EVs: 252 HP / 252 Def / 4 SpD  
Bold Nature  
IVs: 0 Atk  
- Light Screen  
- Rest  
- Wish  
- Protect  

Gholdengo @ Choice Scarf  
Ability: Good as Gold  
Tera Type: Steel  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Flash Cannon  
- Shadow Ball  
- Make It Rain  
- Reflect  
'''


async def play():
    # No authentication required
    my_account_config = AccountConfiguration("pokemonrlbotcs486", None)
    # player = RandomPlayer(account_configuration=my_account_config, battle_format="gen9ou", team=ou_team)
    player = RandomPlayer(account_configuration=my_account_config)
    # team = Teambuilder.parse_showdown_team(ou_team)
    print("Starting bot...")
    # print(str(team)[1:-1])
    await player.accept_challenges(None, 1)

if __name__ == "__main__":
    asyncio.run(play())
