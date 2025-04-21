from poke_env import AccountConfiguration, ServerConfiguration
from bot.bot import RLBot
import bot.teams as teams
from bot.bot_mapping import EnvironmentEncoder
from encoding.autoencoder import Autoencoder
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
import matplotlib.pyplot as plt
import pickle

from trainer.actorcritic import create_agent
from trainer.trainer import Trainer
import asyncio
import torch
import torch.optim as optim
from copy import deepcopy
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluate',
                    action='store_true')
    parser.add_argument('-t', '--train',
                    action='store_true')
    parser.add_argument('-c', '--checkpoint')
    args = parser.parse_args()

    NETWORK_HIDDEN_DIM = 512
    NETWORK_DROPOUT = 0.2
    LEARNING_RATE = 0.0001

    trainer = Trainer(battles_per_epoch=50)
    encoder = EnvironmentEncoder()

    if args.train:
        print("Starting Training")
        if args.checkpoint:
            decision_network = torch.load(args.checkpoint, weights_only=False)
        else:
            decision_network = create_agent(NETWORK_HIDDEN_DIM, NETWORK_DROPOUT).cpu()

        torch.save(decision_network, "bot_data/main_model.pth")

        optimizer = optim.Adam(decision_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        
        bot = RLBot("rlbotcs486", "gen7randombattle", None, decision_network, encoder)
        opp = SimpleHeuristicsPlayer(AccountConfiguration("rloppcs486", None), battle_format="gen7randombattle",  max_concurrent_battles=10)
        # opp = RLBot("rloppcs486", "gen7randombattle", None, decision_network, encoder)

        losses, evals, steps = trainer.run_epochs(bot, opp, optimizer, 300, set_weights_interval=10, checkpoint_interval=5, evaluation_interval=5)

        with open("training_results.pkl", "wb") as f:
            pickle.dump({"loss": losses, "eval": evals, "steps": steps}, f)
    
    elif args.evaluate:
        print("Starting Evaluation")
        if args.checkpoint:
            decision_network = torch.load(args.checkpoint, weights_only=False)
        else:
            decision_network = torch.load("bot_data/checkpoints/heuristic_play_checkpoint.pth", weights_only=False)
            
            bot = RLBot("rlbotcs486", "gen7randombattle", None, decision_network, encoder)
            opp = SimpleHeuristicsPlayer(AccountConfiguration("rloppcs486", None), battle_format="gen7randombattle",  max_concurrent_battles=10)
            opp2 = RandomPlayer(AccountConfiguration("rloppcs4862", None), battle_format="gen7randombattle",  max_concurrent_battles=10)
            opp3 = RLBot("rloppcs4863", "gen7randombattle", None, decision_network, encoder)
            print("Running SimpleHeuristicsPlayer")
            opp_res_1 = trainer.evaluate(bot, opp, 1000)
            print("Running RandomPlayer")
            opp_res_2 = trainer.evaluate(bot, opp2, 1000)
            print("Running RLBot")
            opp_res_3 = trainer.evaluate(bot, opp3, 1000)

            print("SimpleHeuristicsPlayer:")
            print(f"{opp_res_1[0] * 100}% WR, {opp_res_1[1].item()} average discounted reward per turn")
            print("RandomPlayer:")
            print(f"{opp_res_2[0] * 100}% WR, {opp_res_2[1].item()} average discounted reward per turn")
            print("Self:")
            print(f"{opp_res_3[0] * 100}% WR, {opp_res_3[1].item()} average discounted reward per turn")
