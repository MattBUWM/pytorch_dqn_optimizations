import json

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    model_path = 'trained/base_gelu'
    statistics = json.load(open(model_path + '/statistics.json'))
    
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
                (0, 0.25),
                (0, 200),
                (0, 10000),
                (0, 0.05),
                (0, 200)
            ]

    ):
        fig = plt.figure()
        plt.xlim((1, 100))
        plt.ylim(z)
        plt.title(y)
        plt.plot(range(1, len(statistics[x])+1), statistics[x])
        plt.show()
