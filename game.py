import gym, obstacle_env, random

if __name__ == '__main__':
    env = gym.make("obstacle-v0")
    env.reset()
    done = False
    while not done:
        env.render()
        action =  random.randint(0,4)# Your agent code here
        obs, reward, done, info = env.step(action)
        
    env.close()