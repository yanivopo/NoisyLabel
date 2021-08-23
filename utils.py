import numpy as np
from matplotlib import pyplot as plt


def create_pair(imgs, labels, example_numer=5000):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    img_pairs = []
    pair_labels = []
    real_label = []
    num_imgs = imgs.shape[0]
    count = 0
    while count < example_numer:
        first_idx, second_idx = np.random.choice(np.arange(num_imgs), 2)
        if labels[first_idx] == labels[second_idx]:
            continue
        img_pairs.append([imgs[first_idx], imgs[second_idx]])
        dict_temp = \
            {i: 1 if j in (labels[first_idx][0], labels[second_idx][0]) else 0 for j, i in enumerate(class_names)}
        pair_labels.append(dict_temp)
        real_label.append([labels[first_idx], labels[second_idx]])
        count += 1
    return np.array(img_pairs), pair_labels, np.array(real_label).reshape(-1, 1)


def plot_example(images, label, save=None):
    plt.figure(figsize=(12, 12))
    for i in range(3):
        for j in range(2):
            plt.subplot(3, 2, (i*2) + j+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i][j])
        plt.xlabel(label[i], fontsize=9, horizontalalignment='right')
    if save:
        plt.savefig(save)
    # plt.show()

