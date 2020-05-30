from sklearn.calibration import calibration_curve
import pickle as pk

from matplotlib import pyplot as plt


if __name__ == '__main__':
    with open('../Datasets/BaseDataset/train_val_test/prediction_val.pickle', 'rb') as file:
        pred_val_saved = pk.load(file)

    with open('../Datasets/BaseDataset/train_val_test/val_y.pickle', 'rb') as file:
        y_val = pk.load(file)

    y_true, y_pred = calibration_curve(y_val.values[:, 0], pred_val_saved[:, 0])

    plt.scatter(y_true, y_pred)
    plt.plot([[0, 0], [1, 1]])

    plt.show()
