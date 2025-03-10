import numpy as np
import torch
from environment.mnist_env.marl_mnist import MarlMNIST


class AgentNetwork:
    def __init__(self, env, agents=None):
        if agents is None:
            self.agents = []
        self.env = env
        self.agents = agents

    def sample(self, horizon, policies):
        """
        Sample a rollout from the agent.

        Arguments:
          horizon: (int) the length of the rollout
          policy: the policy that the agent will use for actions
        """
        assert len(self.agents) == len(policies)

        rewards = []
        states, actions, reward_sum, done = [self.env.reset()], [], 0, False

        # policy.reset()
        for t in range(horizon):
            agent_actions = []
            for i, policy in enumerate(policies):
                agent_actions.append(policy.act(states[t], t)) # TODO: update for all agents
                # TODO: UPDATE ALL THE BELOW FOR ALL AGENTS
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
