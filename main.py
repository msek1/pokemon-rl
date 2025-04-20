from poke_env import AccountConfiguration, ServerConfiguration
from bot.bot import RLBot
import bot.teams as teams
from encoding.autoencoder import Autoencoder
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from trainer.actorcritic import create_agent
from trainer.trainer import Trainer
import asyncio
import torch.optim as optim
from torch.utils.data import TensorDataset 
import torch

if __name__ == "__main__":
    NETWORK_HIDDEN_DIM = 512
    NETWORK_DROPOUT = 0.2
    decision_network = create_agent(NETWORK_HIDDEN_DIM, NETWORK_DROPOUT)
    
    bot = RLBot("pokemonrlbotcs486", "gen7ou", "gen_7_team", decision_network)
    
    opp = SimpleHeuristicsPlayer(AccountConfiguration("rlbotopp", None), battle_format="gen7ou",  max_concurrent_battles=20, team=teams.teams["gen_7_team"])
    
    trainer = Trainer()
    # asyncio.run(trainer.make_players_play(bot, opp, 20))

    battle_returns_advantages = trainer.run_epoch(bot, opp)
    # battle ids are not needed at this point
    states = []
    actions = []
    actions_log_probs = []
    values = []
    returns = []
    advantages = []
    for battle in bot.battle_data:
        data = bot.battle_data[battle]
        for entry in data:
            states.append(entry[0])
            actions.append(entry[1].item())
            actions_log_probs.append(entry[2].item())
            values.append(entry[3].item())

        ra = battle_returns_advantages.get(battle)
        returns.append(ra[0])
        advantages.append(ra[1])
    returns = torch.cat(returns)
    advantages = torch.cat(advantages)
    states = torch.stack(states) 
    actions = torch.tensor(actions)
    actions_log_probs = torch.tensor(actions_log_probs)
    values = torch.tensor(values)

    results_dataset = TensorDataset(
        states.detach(),
        actions.detach(),
        actions_log_probs.detach(),
        advantages.detach(),
        returns.detach()
    )

    PPO_STEPS = 8
    N_TRIALS = 100
    EPSILON = 0.2
    ENTROPY_COEFFICIENT = 0.01
    LEARNING_RATE = 0.001
    EPSILON = 0.2
    optimizer = optim.Adam(decision_network.parameters(), lr=LEARNING_RATE)
    policy_loss, value_loss = trainer.update_policy(
        agent=decision_network,
        training_results_dataset=results_dataset,
        ppo_steps=PPO_STEPS,
        optimizer=optimizer,
        entropy_coefficient=ENTROPY_COEFFICIENT,
        epsilon=EPSILON
    )
    print(policy_loss)
    print(value_loss)


                        