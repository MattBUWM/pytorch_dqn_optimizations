import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

sizes = [92296, 23223, 184591, 184588, 46394, 369045, 46397, 92741]
labels = ['base_gelu', 'convolution_gelu', 'noisy_gelu', 'dueling_gelu', 'convolution_dueling_gelu', 'dueling_noisy_gelu', 'convolution_noisy_gelu', 'all_gelu']

if __name__ == '__main__':
    fig = plt.figure(figsize=(8, 6))
    plt.bar(labels, sizes)
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d KB'))
    plt.title('Rozmiary pojedyńczych punktów kontrolnych modeli')
    plt.tight_layout()
    fig.savefig('plots/checkpoint_file_sizes.jpg')