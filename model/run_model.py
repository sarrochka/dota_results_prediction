import tensorflow as tf
import numpy as np
import pickle as pk

from model.nn import ResultPredictingModel

if __name__ == '__main__':
    with open('../Datasets/BaseDataset/train_val_test/val_x.pickle', 'rb') as file:
        val_x = pk.load(file)
    with open('../Datasets/BaseDataset/train_val_test/val_y.pickle', 'rb') as file:
        val_y = pk.load(file)

    n_x = val_x.shape[1]
    n_y = val_y.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, n_x, 1], name="x")
    y = tf.placeholder(tf.float32, shape=[None, n_y], name="y")

    h = [500, 100, 50]
    model = ResultPredictingModel(x_=x, y_=y, h_list=h, cost_weigths_= np.asarray([0.5, 0.5]))
    model.initialize()
    saver = tf.train.Saver()

    with open('../Datasets/BaseDataset/train_val_test/prediction_val.pickle', 'rb') as file:
        pred_val_saved = pk.load(file)

    with tf.Session() as sess:
        saver.restore(sess, "model.ckpt")
        pred_val = sess.run(model.prediction(), feed_dict={x: np.expand_dims(val_x, axis=-1), y: val_y})

    print(np.all(pred_val_saved == pred_val))
