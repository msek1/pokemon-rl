from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.environment.observation import Observation
from poke_env.environment.battle import AbstractBattle
from bot.bot import RLBot
from bot.bot_mapping import EnvironmentEncoder
from trainer.actorcritic import ActorCritic
from trainer.reward_policy import RewardPolicy
import asyncio
from copy import deepcopy
from tqdm import tqdm
import time
from multiprocessing import Process
import os
import pickle

EVAL_BATTLES = 100
NUM_PROCESSES = 5
RANDOM_BATTLE_FORMAT = "gen7randombattle"

BOT_NET_FILE = "bot_data/main_model.pth"
OPP_NET_FILE = "bot_data/opp_model.pth"

FAINT_EVENT = "faint"
DAMAGE_EVENT = "-damage"
HEAL_EVENT = "-heal"

import torch
from torch import distributions
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def run_battles(i, n):
    trainer = Trainer()
    net = torch.load(BOT_NET_FILE, weights_only=False)
    opp_net = torch.load(OPP_NET_FILE, weights_only=False)
    encoder = EnvironmentEncoder()
    # opp = SimpleHeuristicsPlayer(AccountConfiguration(f"rlbotopp{i+1}", None), battle_format="gen7randombattle",  max_concurrent_battles=20)
    trainer.create_agent_and_run_battles(f"trainbotwkr{i+1}", f"rlbotopp{i+1}", net, opp_net, encoder, n)

def merge_dicts(d1, d2):
    for (k,v) in d2.items():
        d1[k] = v
    return d1

class Trainer:
    def __init__(self,
                 lr = 0.001, gamma = 0.98, battles_per_epoch = 1,
                 ppo_steps = 8, clip_epsilon = 0.2, entropy_coefficient = 0.01,
                 reward_policy: RewardPolicy=RewardPolicy()) -> None:
        self.lr = lr
        self.gamma = gamma
        self.battles_per_epoch = battles_per_epoch
        self.ppo_steps = ppo_steps
        self.clip_epsilon = clip_epsilon
        self.entropy_coefficient = entropy_coefficient
        self.reward_policy = reward_policy
    
    def run_epochs(self, p1: RLBot, p2: Player, optimizer: optim.Optimizer, num_epochs: int = 1,  set_weights_interval: int = 0, evaluation_interval: int = 0, checkpoint_interval:int = 0):
        eval_results = []
        loss_results = []
        num_steps = []
        p2.decision_network = deepcopy(p1.decision_network)

        for e in tqdm(range(1,num_epochs + 1)):
            aloss, vloss, steps = self.run_epoch(p1, p2, optimizer)
            loss_results.append((aloss, vloss))
            num_steps.append(steps)
            p1.clear_battle_data()
            p1.prev_battle_obs_count = {}

            if set_weights_interval > 0 and e % set_weights_interval == 0 and p2 is RLBot:
                p2.decision_network = deepcopy(p1.decision_network)
            
            if checkpoint_interval > 0 and e % checkpoint_interval == 0:
                torch.save(p1.decision_network, f"bot_data/checkpoints/model_{time.time()}.pth")

            if evaluation_interval > 0 and e % evaluation_interval == 0:
                p1.clear_battle_data()
                with torch.no_grad():
                    asyncio.run(self.make_players_play(p1, p2, EVAL_BATTLES))
                eval_results.append(self.calculate_eval_results(p1))
            
            if p2 is RLBot:
                p2.clear_battle_data()

        return loss_results, eval_results if len(eval_results) > 0 else None, num_steps

    def run_epoch(self, p1: RLBot, p2: Player, optimizer: torch.optim.Optimizer):
        asyncio.run(self.make_players_play(p1, p2, self.battles_per_epoch))
        battle_returns_advantages = self.calculate_all_returns_advantages(p1.battles, p1.battle_data)

        num_steps = sum([len(v) for v in p1.battle_data.values()])

        # battle ids are not needed at this point
        states = []
        actions = []
        actions_log_probs = []
        values = []
        returns = []
        advantages = []
        for battle in p1.battle_data:
            data = p1.battle_data[battle]
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

        policy_loss, value_loss = self.update_policy(
            agent=p1.decision_network,
            training_results_dataset=results_dataset,
            ppo_steps=self.ppo_steps,
            optimizer=optimizer,
            entropy_coefficient=self.entropy_coefficient,
            epsilon=self.clip_epsilon
        )

        return policy_loss, value_loss, num_steps
    
    def run_parallel_battles(self):
        procs = []
        for i in range(NUM_PROCESSES):
            p = Process(target=run_battles, args = (i, (self.battles_per_epoch // NUM_PROCESSES) + 1))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()

    def read_battle_data_and_clean(self):
        files = os.listdir("bot_data/tmp/")
        battles = {}
        battle_data = {}
        for fname in files:
            with open(f"bot_data/tmp/{fname}", "rb") as f:
                d = pickle.load(f)
            battles = merge_dicts(battles, d["battles"])
            battle_data = merge_dicts(battle_data, d["states"])
            os.remove(f"bot_data/tmp/{fname}")
        return battles, battle_data

    def calculate_all_returns_advantages(self, battles, battle_data):
        battle_returns_advantages = {}
        for (tag, battle) in battles.items():
            battle_data_episode = battle_data[tag]
            values = [step[3].item() for step in battle_data_episode]
            battle_returns_advantages[tag] = self.calculate_reward_for_battle(battle, values, len(battle_data[tag]))

        return battle_returns_advantages

    def calculate_reward_for_battle(self, battle: AbstractBattle, values, n_states):
        rewards = [
            self.calculate_reward_for_turn(battle.observations[i], battle.player_role)
            for i in range(1, len(battle.observations))
        ]
        rewards = rewards[:n_states]
        if battle.won:
            rewards[-1] += self.reward_policy.win_reward
        else:
            rewards[-1] -= self.reward_policy.win_reward
        
        return self.do_gae_episode(rewards=rewards, values=values)

    def calculate_reward_for_turn(self, turn: Observation, player_role: str):
        turn_reward = 0
        for evt in turn.events:
            if len(evt) <= 2:
                continue
            evt_type = evt[1]
            evt_target = evt[2]
            
            is_player_target = evt_target.startswith(player_role)

            if evt_type == FAINT_EVENT:
                if is_player_target:
                    turn_reward -= self.reward_policy.faint_reward
                else:
                    turn_reward += self.reward_policy.faint_reward
            elif evt_type == DAMAGE_EVENT or evt_type == HEAL_EVENT:
                fraction = evt[3].split()[0].split("/")
                if len(fraction) == 1:
                    health_prop = 0
                else:
                    health_prop = float(fraction[0]) / float(fraction[1])

                if is_player_target:
                    previous_health = turn.active_pokemon.current_hp_fraction
                else:
                    previous_health = turn.opponent_active_pokemon.current_hp_fraction
                health_diff = previous_health - health_prop

                if evt_type == DAMAGE_EVENT:
                    if is_player_target:
                        turn_reward -= health_diff * self.reward_policy.damage_proportion * self.reward_policy.own_damage_factor
                    else:
                        turn_reward += health_diff * self.reward_policy.damage_proportion
                else:
                    if is_player_target:
                        turn_reward += -health_diff * self.reward_policy.healing_proportion
                    else:
                        turn_reward -= -health_diff * self.reward_policy.healing_proportion * self.reward_policy.own_damage_factor**3
        return turn_reward

    async def make_players_play(self, p1: RLBot, p2: Player, num_battles: int):
        await asyncio.gather(
            p1.accept_challenges(None, num_battles),
            self.delayed_challenge(p1, p2, num_battles)
        )

    async def delayed_challenge(self, p1: RLBot, p2: Player, num_battles: int):
        time.sleep(0.5)
        await p2.send_challenges(p1.name, num_battles)
    
    def calculate_eval_results(self, p1: RLBot): # -> win percentage, average reward
        w = 0
        for b in p1.battles.values():
            if b.won: w+=1
        rewards = self.calculate_all_discounted_reward(p1)

        return (w / len(p1.battles), sum(rewards) / len(rewards))
    
    def do_gae_episode(self, rewards, values, lam=0.95):
        advantages = []
        gae = 0
        values = values + [0]  # append 0 for V(s_{t+1}) at end

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * lam * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return (returns, advantages)

    def calculate_surrogate_loss(
        actions_log_probability_old,
        actions_log_probability_new,
        epsilon,
        advantages):
        advantages = advantages.detach()
        policy_ratio = (
                actions_log_probability_new - actions_log_probability_old
                ).exp()
        surrogate_loss_1 = policy_ratio * advantages
        surrogate_loss_2 = torch.clamp(
                policy_ratio, min=1.0-epsilon, max=1.0+epsilon
                ) * advantages
        surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
        return surrogate_loss
    
    def calculate_losses(
        surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
        entropy_bonus = entropy_coefficient * entropy
        policy_loss = -(surrogate_loss + entropy_bonus).sum()
        value_loss = f.smooth_l1_loss(returns, value_pred)
        return policy_loss, value_loss
    

    def update_policy(self, agent, training_results_dataset, ppo_steps, optimizer, entropy_coefficient, epsilon):        
        BATCH_SIZE = 128
        total_policy_loss = 0
        total_value_loss = 0
        
        batch_dataset = DataLoader(
            training_results_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False)
        
        for _ in range(ppo_steps):
            for (states, actions, actions_log_probability_old, advantages, returns) in batch_dataset:
                action_pred, value_pred = agent(states)
                value_pred = value_pred[:, 0] # the predicted values across the batch

                action_prob = f.softmax(action_pred, dim=-1) # the action distributions across the batch

                action_distribution = distributions.Categorical(action_prob)
                action_log_distribution = action_distribution.log_prob(actions) # the current log-distribution of actions taken
                entropy = action_distribution.entropy()
                surrogate_loss = Trainer.calculate_surrogate_loss(
                    actions_log_probability_old,
                    action_log_distribution,
                    epsilon,
                    advantages)
                
                policy_loss, value_loss = Trainer.calculate_losses(
                    surrogate_loss,
                    entropy,
                    entropy_coefficient,
                    returns,
                    value_pred)
                
                optimizer.zero_grad()
                total_loss = policy_loss + value_loss
                total_loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

    def create_agent_and_run_battles(self, name: str, opp_name: str, net: ActorCritic, opp_net: ActorCritic, encoder: EnvironmentEncoder, n: int):
        bot = RLBot(name, RANDOM_BATTLE_FORMAT, None, net, encoder)
        opp = RLBot(opp_name, RANDOM_BATTLE_FORMAT, None, opp_net, encoder)
        asyncio.run(self.make_players_play(bot, opp, n))
        bot.write_out_battles()

