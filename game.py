import gym, obstacle_env, random
from agent import model,batch_size, model_save_path
import numpy as np
import torch
render = False

if __name__ == '__main__':
    env = gym.make("obstacle-v0")
    obs = env.reset()
    done = False
    xs_episode = []
    ys_episode = []
    xs = []
    ys = []
    rewards = []
    rewards_episode = []
    rewards = []
    reward_sum = 0
    episode_number = 0
    running_reward = None
    optimizer = torch.optim.RMSprop(model.parameters())
    while True:
        if render:
            env.render()
        with torch.no_grad:
            action_probs =  model(obs)
        prob_thresh = np.random.uniform()
        action = np.argmax(action_probs)
        if prob_thresh < 0.2:
            action = env.action_space.sample() #random action for exploration
        xs_episode.append(obs)
        y = 2 if action == 2 else 0 #fake label
        ys_episode.append(y)
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        rewards_episode.append(reward)
        if done:
            #episode finished
            episode_number += 1
            xs.append(xs_episode)
            ys.append(ys)
            xs_episode = []
            ys_episode = []
            discounted_rewards = model.discount_rewards(rewards_episode)
            discounted_rewards = np.array(discounted_rewards)
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            rewards.append(discounted_rewards)
            rewards_episode = []

            
            if episode_number % batch_size == 0:
                #reset training data
                optimizer.zero_grad()
                loss = None # To do
                loss.backward()
                for param in model.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            obs = env.reset()
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if episode_number % 100 == 0:
                 torch.save(model.state_dict(), model_save_path)
            reward_sum = 0
    env.close()