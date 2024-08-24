import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

path = 'trained/'
models = ['base_gelu', 'convolution_gelu', 'noisy_gelu', 'dueling_gelu', 'convolution_dueling_gelu', 'dueling_noisy_gelu', 'convolution_noisy_gelu', 'all_gelu']
mean_time = []

if __name__ == '__main__':
    for x in models:
        statistics = json.load(open(path + x + '/statistics.json'))
        sum_time = sum(statistics['episode_training_time'])
        sum_steps_count = sum(statistics['episode_steps_count'])
        mean_time.append(sum_time / sum_steps_count)
    fig = plt.figure(figsize=(8, 6))
    plt.bar(models, mean_time)
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%0.2f s'))
    plt.title('średnie czasy trwania kroku podczas trenowania modelu')
    plt.tight_layout()
    fig.savefig('plots/checkpoint_step_mean_time.jpg')
