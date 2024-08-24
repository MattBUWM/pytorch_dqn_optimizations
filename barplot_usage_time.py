import json
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import envs.SuperJetPakEnv
from models.BaseModel import BaseModel
from models.DQN import DQN

path = 'trained/'
models = [
    ('base_gelu', True),
    ('convolution_gelu', False),
    ('noisy_gelu', True),
    ('dueling_gelu', True),
    ('convolution_dueling_gelu', False),
    ('dueling_noisy_gelu', True),
    ('convolution_noisy_gelu', False),
    ('all_gelu', False)
]
model_names = []
mean_time = []

if __name__ == '__main__':
    for x in models:
        print(x[0])
        env = envs.SuperJetPakEnv.SuperJetPakEnv('roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc',
                                                 'roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc.state',
                                                 force_gbc=False,
                                                 ticks_per_action=4,
                                                 force_discrete=True,
                                                 flatten=x[1],
                                                 grayscale=True)
        model = DQN.load('trained/' + x[0])
        observation, info = env.reset()
        measured_time = 0
        for _ in range(5000):
            start_time = time.time()
            action = model.predict(observation, env)
            observation, reward, terminated, truncated, info = env.step(action)
            measured_time += time.time() - start_time
        env.close()
        model_names.append(x[0])
        mean_time.append(measured_time/5000)
    fig = plt.figure(figsize=(8, 6))
    plt.bar(model_names, mean_time)
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%0.4f s'))
    plt.title('średnie czasy trwania kroku podczas używania modelu')
    plt.tight_layout()
    fig.savefig('plots/checkpoint_step_mean_time_during_usage.jpg')
