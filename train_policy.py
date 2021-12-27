from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from virtual_env import get_env_instance


if __name__ == '__main__':
    env = get_env_instance('./data/user_states_by_day.npy', './data/venv.pkl')
    #model = TD3("MlpPolicy", env, batch_size=420, verbose=1, tensorboard_log='./data/logs')
    #model = PPO("MlpPolicy", env, n_steps=840, batch_size=420, verbose=1, tensorboard_log='./data/logs')
    #model = DQN("MlpPolicy", env,  batch_size=420, verbose=1, tensorboard_log='./data/logs')
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log='./data/logs')
    checkpoint_callback = CheckpointCallback(save_freq=8e4, save_path='./data/model_checkpoints')
    model.learn(total_timesteps=int(8e6), callback=[checkpoint_callback])
