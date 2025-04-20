from poke_env.player import Player
from poke_env.environment.observation import Observation
from poke_env.environment.battle import AbstractBattle
from bot.bot import RLBot
from trainer.reward_policy import RewardPolicy
import asyncio
from copy import deepcopy
from tqdm import tqdm

EVAL_BATTLES = 100

FAINT_EVENT = "faint"
DAMAGE_EVENT = "-damage"
HEAL_EVENT = "-heal"

from torch.utils.data import TensorDataset, DataLoader
from torch import distributions
import torch.nn.functional as f
import torch
import torch.optim as optim

class Trainer:
    def __init__(self, lr = 0.001, gamma = 0.98, battles_per_epoch = 1, reward_policy: RewardPolicy=RewardPolicy()) -> None:
        self.lr = lr
        self.gamma = gamma
        self.battles_per_epoch = battles_per_epoch
        self.reward_policy = reward_policy
    
    def run_epochs(self, p1: RLBot, p2: Player, num_epochs: int = 1,  set_weights_interval: int = 0, evaluation_interval: int = 0):
        eval_results = []
        for e in tqdm(range(1,num_epochs + 1)):
            self.run_epoch(p1, p2)
            if set_weights_interval > 0 and e % set_weights_interval == 0 and p2 is RLBot:
                p2.decision_network = deepcopy(p1.decision_network)
            
            if evaluation_interval > 0 and e % evaluation_interval == 0:
                p1.clear_battle_data()
                asyncio.run(self.make_players_play(p1, p2, EVAL_BATTLES))
                eval_results.append(self.calculate_eval_results(p1))

            if p2 is RLBot:
                p2.clear_battle_data()

        return eval_results if len(eval_results) > 0 else None


    def run_epoch(self, p1: RLBot, p2: Player):
        asyncio.run(self.make_players_play(p1, p2, self.battles_per_epoch))

        battle_returns_advantages = self.calculate_all_returns_advantages(p1)

        states = {
            tag: [t[0] for t in data] for (tag,data) in p1.get_battle_data().items()
        }
        actions = {
            tag: [t[1] for t in data] for (tag,data) in p1.get_battle_data().items()
        }

        return battle_returns_advantages
    
    def calculate_all_returns_advantages(self, p: RLBot):
        battle_returns_advantages = {}
        for (tag, battle) in p.battles.items():
            battle_data_episode = p.battle_data[tag]
            values = [step[3].item() for step in battle_data_episode]
            battle_returns_advantages[tag] = self.calculate_reward_for_battle(battle, values)

        return battle_returns_advantages

    def calculate_reward_for_battle(self, battle: AbstractBattle, values):
        rewards = [
            self.calculate_reward_for_turn(battle.observations[i], battle.player_role)
            for i in range(1, len(battle.observations))
        ]
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
            p2.send_challenges(p1.name, num_battles)
        )
    
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


