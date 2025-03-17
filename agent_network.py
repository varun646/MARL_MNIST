import numpy as np
import torch
from environment.mnist_env.marl_mnist import MarlMNIST
from agent import Agent


class AgentNetwork:
    def __init__(self, env, agents=None):
        self.agents = agents

        if agents is None:
            self.agents = []

        self.env = env

    def sample_for_agent(self, agent_num, horizon, policy):
        assert agent_num < len(self.agents)
        assert agent_num > 0

        rewards = []
        states, actions, reward_sum, done = [self.env.reset()], [], 0, False

        agent = self.agents[agent_num]
        policy.reset()
        for t in range(horizon):
            actions.append(policy.act(states[t], t))
            state, reward, done, info = self.env.step(actions[t])
            states.append(state)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

        # print("Rollout length: %d,\tTotal reward: %d,\t Last reward: %d" % (len(actions), reward_sum), reward)

        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }


    def sample(self, horizon, policies):
        """
        Sample a rollout from the agent.

        Arguments:
          horizon: (int) the length of the rollout
          policies: the policies that the agent will use for actions
        """
        assert len(self.agents) == len(policies)

        rewards = []
        states, actions, reward_sum, done = [self.env.reset()], [], 0, False

        # policy.reset()
        # for i in range(len(self.agents)):
        obs = []
        ac = []
        reward_sum = []
        rewards = []


        for t in range(horizon):
            for i in range(len(self.agents)):
                agent_sample = self.sample_for_agent(i, horizon=1, policy=policies[i])
                obs.append(agent_sample["obs"])
                ac.append(agent_sample["ac"])
                reward_sum.append(agent_sample["reward_sum"])
                rewards.append(agent_sample["rewards"])

        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }


class RandomPolicyWithMessaging:
    def __init__(self, action_dim: int, num_agents: int, agent_network: AgentNetwork):
        self.action_dim = 2
        self.num_agents = num_agents
        self.agent_network = agent_network

    def reset(self):
        pass

    def act_with_messages(self, messages: list[int]):
        # TODO: include messages in action
        print(messages)
        return np.random.binomial(n=self.action_dim, p=0.5, size=self.num_agents)

    def act(self):
        # TODO: add logic for communication with neighbors
        for agent in self.agent_network.agents:
            agent_action = self.act_with_messages(agent.messages)

        return np.random.binomial(n=self.action_dim, p=0.5, size=self.num_agents)

class RandomPolicy:
    def __init__(self, action_dim: int, num_agents=int):
        self.action_dim = 2
        self.num_agents = num_agents

    def reset(self):
        pass

    def act(self):
        return np.random.binomial(n=self.action_dim, p=0.5, size=self.num_agents)


if __name__ == "__main__":
    env = MarlMNIST()
    policy = RandomPolicy(2)
    agent = Agent(env)
    for _ in range(5):
        agent.sample(20, policy)
