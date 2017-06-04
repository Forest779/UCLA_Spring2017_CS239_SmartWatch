import matplotlib.pyplot as plt
import numpy as np

no_shuffle = [1,0.75,0.5,0.25]
shuffle = [0.25,0.5,0.75,1]

def draw_shuffle_figure(no_shuffle,shuffle):

    #list: 1)acc, 2)precision, 3)recall, 4)F1 score

    if len(no_shuffle) != 4 or len(shuffle) != 4:
        raise ValueError, "Input is wrong"

    else:
        fig, ax = plt.subplots()
        index = np.arange(len(shuffle))


    # create plot

        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, no_shuffle, bar_width,
                        alpha=opacity,
                        color='b',
                        label='without shuffle')

        rects2 = plt.bar(index + bar_width, shuffle, bar_width,
                        alpha=opacity,
                        color='g',
                        label='shuffle')

        plt.xlabel('Metrics')
        plt.ylabel('Scores')
        plt.title('Metrics without shuffle vs shuffle')
        plt.xticks(index + bar_width, ('Acc', 'Prec', 'Rec', 'F1'))
        plt.legend()

        plt.tight_layout()
        plt.savefig("shuffle.png")
        plt.show()

if __name__ == '__main__':
    draw_shuffle_figure(no_shuffle,shuffle)
