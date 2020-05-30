import tensorflow as tf
import numpy as np
import pickle as pk

from Datasets.BaseDataset.reader import DataReader


class ResultPredictingModel:
    def __init__(self, y, h_list, cost_weigths):
        super().__init__(y)
        self.h_list = h_list
        self.cost_weights = cost_weigths
        self.n_y = y.shape[1]

        self._logits = None
        self._prediction = None
        self._cost = None
        self._accuracy = None
        self._optimize = None

    def initialize(self):
        self.prediction()
        self.optimize()

    def logits(self):
        if self._logits is None:
            with tf.name_scope('LSTM'):
                lstm_1 = tf.keras.layers.LSTM(128)(x)

            with tf.name_scope('FC'):
                fc_1 = tf.keras.layers.Dense(self.h_list[0], activation='relu')(lstm_1)
                fc_2 = tf.keras.layers.Dense(self.h_list[1], activation='relu')(fc_1)

            with tf.name_scope('Logits'):
                self._logits = tf.keras.layers.Dense(self.h_list[2], activation='relu')(fc_2)

        return self._logits

    def prediction(self):
        if self._prediction is None:
            with tf.name_scope('Output'):
                self._prediction = tf.keras.layers.Dense(self.n_y, activation='softmax')(self.logits())
        return self._prediction

    def cost(self):
        if self._cost is None:
            with tf.name_scope('Cost'):
                self._cost = tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=self.prediction(),
                                                                      pos_weight=self.cost_weights)
        return self._cost

    def accuracy(self):
        if self._accuracy is None:
            correct_prediction = tf.equal(tf.argmax(self.prediction(), 1, name='ArgmaxPred'),
                                          tf.argmax(self.y, 1, name='YPred'),
                                          name='CorrectPred')
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name='CastCorrectPred'),
                                            name='Accuracy')
        return self._accuracy

    def optimize(self):
        if self._optimize is None:
            self._optimize = tf.train.AdagradOptimizer(lr, name='Optimizer').minimize(self.cost())
        return self._optimize


if __name__ == '__main__':
    feature_cols = ['dire_score', 'radiant_score', 'duration', 'patch', 'region', 'radiant_team_id', 'dire_team_id',
                    'players_radiant_id', 'players_dire_id', 'radiant_team_name', 'dire_team_name']
    y_cols = ['radiant_win']
    x_cols = ['avg_dire_score', 'avg_radiant_score', 'avg_duration', 'patch', 'region']

    # if they are used as input
    # x_cols += ['radiant_team_id', 'dire_team_id']
    # x_cols += [f'radiant_player_{j}' for j in range (1, 6)] + [f'dire_player_{j}' for j in range(1, 6)]

    data_reader = DataReader('Datasets/BaseDataset/dota2_dataset.pickle', feature_cols, y_cols, x_cols)
    data_reader.read_preprocessed('Datasets/BaseDataset/dota2_dataset_preprocessed.pickle')

    train_x, train_y, val_x, val_y, test_x, test_y = data_reader.get_train_val_test()

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    n_x = train_x.shape[1]
    n_y = train_y.shape[1]

    tf.reset_default_graph()

    lr = 1e-3

    g = tf.Graph()

    radiant_wr = np.where(data_reader.preprocessed_data[y_cols])[0].shape[0] / \
                 data_reader.preprocessed_data[y_cols].shape[0]

    print(radiant_wr)
    cost_weigths = np.asarray([radiant_wr, 1.-radiant_wr])

    with g.as_default():
        x = tf.placeholder(tf.float32, shape=[None, n_x, 1], name="x")
        y = tf.placeholder(tf.float32, shape=[None, n_y], name="y")
        tf.constant(cost_weigths)
        saver = tf.train.Saver()

    h = [500, 100, 50]

    model = ResultPredictingModel(y=y, h_list=h, cost_weigths=cost_weigths)

    with g.as_default():
        model.initialize()

    sess = tf.InteractiveSession(graph=g)
    init = tf.global_variables_initializer()
    sess.run(init)

    num_epochs = 250

    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            print(epoch)
        _, c = sess.run([model.optimize(), model.cost()], feed_dict={x: np.expand_dims(train_x, axis=-1), y: train_y})

    save_path = saver.save(sess, "/model.ckpt")

    print('Accuracy val: ', model.accuracy().eval({x: np.expand_dims(val_x, axis=-1), y: val_y}))
    prediction = model.prediction().eval({x:  np.expand_dims(val_x, axis=-1), y: val_y})

    print('Prediction: ', prediction)

    with open('train_x.pickle', 'wb') as file:
        pk.dump(train_x, file)

    with open('train_y.pickle', 'wb') as file:
        pk.dump(train_y, file)

    with open('val_x.pickle', 'wb') as file:
        pk.dump(train_x, file)

    with open('val_y.pickle', 'wb') as file:
        pk.dump(train_y, file)

    with open('test_x.pickle', 'wb') as file:
        pk.dump(train_x, file)

    with open('test_y.pickle', 'wb') as file:
        pk.dump(train_y, file)

    with open('prediction.pickle', 'wb') as file:
        pk.dump(prediction, file)

