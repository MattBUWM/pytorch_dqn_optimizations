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
images = {}

if __name__ == '__main__':
    for x in models:
        images[x[0]] = []
        print(x[0])
        env = envs.SuperJetPakEnv.SuperJetPakEnv('roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc',
                                                 'roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc.state',
                                                 force_gbc=False,
                                                 ticks_per_action=4,
                                                 force_discrete=True,
                                                 flatten=x[1],
                                                 grayscale=True)
        model = DQN.load('trained/' + x[0])
        renders = []
        observation, info = env.reset()
        for t in range(200):
            if t % 10 == 0:
                renders.append(env.render(array=True))
            action = model.predict(observation, env)
            observation, reward, terminated, truncated, info = env.step(action)
        env.close()
        fig, axs = plt.subplots(5, int(len(renders)/5), figsize=(7, 9))
        plt.setp(axs, xticks=[], yticks=[])
        fig.suptitle('wizualizacja środowiska podczas użycia modelu ' + x[0])
        count = 0
        for ax in axs.flatten():
            ax.imshow(renders[count], cmap='gray')
            ax.title.set_text('krok ' + str(count*10))
            count += 1
        plt.tight_layout()
        fig.savefig('plots/' + x[0] + '_play_visualisation.jpg')
