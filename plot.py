import matplotlib.pyplot as plt

def epoch():
    # plot performance with respect to epoch
    x = [1, 2, 3, 4, 5]
    F1 = [85.03, 86.84, 87.46, 87.60, 87.73]
    precision = [86.6, 88.73, 89.29, 89.34, 89.41]
    recall = [83.51, 85.02, 85.70, 85.92, 86.12]

    plt.figure()

    plt.subplot(131)
    plt.plot(x, F1)
    plt.yscale('linear')
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.xticks(x)

    for i, f1 in enumerate(F1):
        plt.annotate(round(f1, 2), (i, f1))

    plt.subplot(132)
    plt.plot(x, precision)
    plt.yscale('linear')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.xticks(x)

    for i, p in enumerate(precision):
        plt.annotate(round(p, 2), (i, p))

    plt.subplot(133)
    plt.plot(x, recall)
    plt.yscale('linear')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.xticks(x)

    for i, r in enumerate(recall):
        plt.annotate(round(r, 2), (i, r))

    plt.subplots_adjust(left=0.10, right=0.99,
                        wspace=0.55)

    plt.show()


def eta():
    # plot performance with respect to eta
    x = [0.5, 0.75, 1, 1.25]
    F1 = [87.67, 87.67, 87.46, 87.31]
    precision = [89.36, 89.42, 89.29, 89.20]
    recall = [86.05, 86, 85.70, 85.5]

    plt.figure()

    plt.subplot(131)
    plt.plot(x, F1)
    plt.yscale('linear')
    plt.ylabel('F1 Score')
    plt.xlabel('Eta')
    plt.grid(True)
    plt.xticks(x)

    for i, f1 in enumerate(F1):
        plt.annotate(round(f1, 2), (x[i], f1))

    plt.subplot(132)
    plt.plot(x, precision)
    plt.yscale('linear')
    plt.ylabel('Precision')
    plt.xlabel('Eta')
    plt.grid(True)
    plt.xticks(x)

    for i, p in enumerate(precision):
        plt.annotate(round(p, 2), (x[i], p))

    plt.subplot(133)
    plt.plot(x, recall)
    plt.yscale('linear')
    plt.ylabel('Recall')
    plt.xlabel('Eta')
    plt.grid(True)
    plt.xticks(x)

    for i, r in enumerate(recall):
        plt.annotate(round(r, 2), (x[i], r))

    plt.subplots_adjust(left=0.15, right=0.95,
                        wspace=0.55)

    plt.show()

if __name__=="__main__":
    epoch()
    eta()