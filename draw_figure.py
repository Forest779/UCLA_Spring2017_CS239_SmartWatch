import matplotlib.pyplot as plt


def draw_shuffle_figure(self,no_shuffle,shuffle):

    #list: 1)acc, 2)precision, 3)recall, 4)F1 score

    if len(no_shuffle) != 4 or len(shuffle) != 4:
        raise ValueError, "Input is wrong"

    else:
        for i in range(4):
