import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    path = 'trained/'
    models = ['convolution_dueling_gelu', 'convolution_dueling_gelu_per', 'convolution_dueling_gelu_multistates', 'convolution_dueling_gelu_multistates_per_v2']
    file_prefix = 'per_and_multistates_'
    
    for x, y, z in zip(
            [
                'episode_rewards_mean',
                'episode_rewards_sum',
                'episode_steps_count',
                'episode_loss_mean',
                'episode_training_time'
            ],
            [
                'średnia nagród w epizodzie',
                'suma nagród w epizodzie',
                'ilość kroków w epizodzie',
                'średnie straty w epizodzie',
                'czas trwania epizodu'
            ],
            [
                (-0.02, 0.1),
                (-25, 200),
                (0, 10000),
                (0.0001, 100),
                (0, 200)
            ]

    ):
        fig = plt.figure()
        plt.xlim((1, 100))
        if x == 'episode_loss_mean':
            plt.yscale('log')
        plt.ylim(z)
        plt.title(y)
        for a in models:
            statistics = json.load(open(path + a + '/statistics.json'))
            plt.plot(range(1, len(statistics[x])+1), statistics[x], label=a, linewidth=1)
        plt.legend()
        fig.savefig('plots/' + file_prefix + x + '.jpg')

