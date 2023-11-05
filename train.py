import gym
from agent import *
import warnings
from utils import *
import random

# Suppress all warnings
warnings.filterwarnings("ignore")

def log(score, games_played):
    # log to CSV file
    appendToFile("{},{}".format(score, games_played), "data/score_per_games_played.csv")
    pass

def train(episodes):
    print("Epoch progress: 0%                  ", end='\r')
    env = gym.make("ALE/MsPacman-v5", render_mode='rgb_array', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    
    total_reward = 0
    for ep in range(episodes):
        env.reset(seed=random.randint(0, 10000))
        obs = env.unwrapped.ale.getRAM()
        state = buildStateFromRAM(obs)
        next_is_random = 0
        lives_left = 3
        while True:
            if render:
                game_img = env.render()
                cv2.imshow('MSPACMAN',scaleImage(game_img))
                cv2.waitKey(1)
            if next_is_random > 0:
                next_is_random -= 1
            else:
                action, next_is_random = agent.getAction(state)
            _, real_reward, done, _, _ = env.step(action)
            total_reward += real_reward
            next_obs = env.unwrapped.ale.getRAM()
            state = buildStateFromRAM(next_obs)
            reward = reward_fn(obs, next_obs, real_reward)
            if next_obs[123] < obs[123]:
                lives_left -= 1
                print("Epoch progress: {}%...               ".format(100*(3*ep+(3-lives_left))/(3*episodes)), end='\r')
            agent.update(state, action, reward)
            obs = next_obs
            if done:
                break
        agent.games_played += 1
        
    log(total_reward/episodes, agent.games_played)
    env.close()
    
def generateGameplayVideos(num_games_played):
    print("Recording progress: 0%                  ", end='\r')
    p_random_action = agent.p_random_action
    explore_unseen_states = agent.explore_unseen_states
    agent.p_random_action = 0
    env = gym.make("ALE/MsPacman-v5", render_mode='rgb_array', full_action_space=False, frameskip=1, repeat_action_probability=0, obs_type='ram')
    for i in range(3):
        env.reset(seed=random.randint(0, 10000))
        obs = env.unwrapped.ale.getRAM()
        state = buildStateFromRAM(obs)
        total_reward = 0
        video = []
        next_is_random = 0
        while True:
            game_img = env.render()
            if render:
                cv2.imshow('MSPACMAN',scaleImage(game_img))
                cv2.waitKey(1)
            video.append(game_img)
            if next_is_random > 0:
                next_is_random -= 1
            else:
                action, next_is_random = agent.getAction(state)
            _, real_reward, done, _, _ = env.step(action)
            state = buildStateFromRAM(env.unwrapped.ale.getRAM())
            total_reward += reward_fn(obs, env.unwrapped.ale.getRAM(), real_reward)
            obs = env.unwrapped.ale.getRAM()
            if done:
                break
        print("Recording progress: {}%...               ".format((i+1)/3), end='\r')
        out = cv2.VideoWriter("recordings/{}_games_played_vid_{}.mp4".format(num_games_played, i+1), cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (game_img.shape[1], game_img.shape[0]))
        for frame in video:
            out.write(frame)
    env.close()
    agent.p_random_action = p_random_action
    agent.explore_unseen_states = explore_unseen_states

if __name__ == '__main__':
    try:
        agent = loadFromPickle("data/checkpoints/agent.pkl")
    except:
        agent = makeAgent()

    games_per_epoch = 5
    epochs_per_recording = 20
    render = True
    
    num_epochs = 0
    while True:
        train(games_per_epoch)
        num_epochs += 1
        if num_epochs % epochs_per_recording == 0:
            generateGameplayVideos(num_epochs*games_per_epoch + agent.games_played)
            saveToPickle("data/checkpoints/agent_{}_games.pkl".format(num_epochs*games_per_epoch + agent.games_played), agent)