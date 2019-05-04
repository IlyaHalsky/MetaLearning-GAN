if __name__ == '__main__':
    import numpy as np
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    x1 = [1,4,6]
    y1 = [1,1,1]
    x2 = [1,2,3]
    y2 = [2,2,2]
    # y2 = savgol_filter(y, window_length=351, polyorder=3)
    # z2 = savgol_filter(z, window_length=351, polyorder=3)

    fig, ax = plt.subplots()

    r, = plt.plot(x2, y2, color='r', label='Evolution')
    f, = plt.plot(x1, y1, color='b', label="Meta-GAN")
    # plt.plot(x, z,color='b', linestyle='-', linewidth=1, alpha=.1)
    # f, = plt.plot(x, z2, color='b', label='R/F Fake')
    fig.suptitle('Качество генерации наборов данных')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Время, часов', fontsize=10)
    plt.ylabel('Расстояние Махаланобиса', fontsize=10)
    plt.grid(linestyle='-', alpha=.2)
    plt.legend(handles=[r, f])
    # plt.ylim(top=1.0)
    plt.show()