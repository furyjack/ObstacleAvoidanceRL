import gym, obstacle_env, random
from agent import model,batch_size, model_save_path
import numpy as np
import torch
render = True

if __name__ == '__main__':
    env = gym.make("obstacle-v0")
    obs = env.reset()
    done = False
    rewards = []
    ys = []
    a = []
    reward_sum = 0
    episode_number = 0
    running_reward = None
    loss = 0
    optimizer = torch.optim.RMSprop(model.parameters())
    while True:
        if render:
            env.render()
        with torch.no_grad():
            obs = torch.Tensor(obs)
        action_probs = model(obs)
        prob_thresh = np.random.uniform()
        action = torch.argmax(action_probs).item()
        # if action == 0:
        #     action = 4
        # if prob_thresh < 0.3:
        #     action = env.action_space.sample() + 1
        #     if action > 4:
        #         action = 1#random action for exploration
        ys.append(action)
        a.append(action_probs)
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        rewards.append(reward)
        if done:
            #episode finished
            episode_number += 1

            discounted_rewards = model.discount_rewards(torch.Tensor(rewards))
            discounted_rewards = np.array(discounted_rewards)
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            loss += torch.sum(torch.nn.functional.cross_entropy(torch.stack(a), torch.LongTensor(ys)) * torch.Tensor(discounted_rewards))
            ys = []
            rewards = []
            a = []
            
            if episode_number % batch_size == 0:
                #reset training data
                optimizer.zero_grad()
                #states = torch.Tensor(xs)
                loss.backward()
                loss = 0
                for param in model.parameters():

                    param.grad.data = -1 * param.grad.data
                optimizer.step()

            obs = env.reset()
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if episode_number % 100 == 0:
                 torch.save(model.state_dict(), model_save_path)
            reward_sum = 0
    env.close()