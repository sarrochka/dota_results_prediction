import sys

import tensorflow as tf
import numpy as np
import pickle as pk
import os
import json
os.chdir('../model')

from nn import ResultPredictingModel

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
    model = ResultPredictingModel(x_=x, y_=y, h_list=h, cost_weigths_=np.asarray([0.5, 0.5]))
    model.initialize()
    saver = tf.train.Saver()

    with open('../Datasets/BaseDataset/train_val_test/prediction_val.pickle', 'rb') as file:
        pred_val_saved = pk.load(file)

    with tf.Session() as sess:
        saver.restore(sess, "model.ckpt")
        pred_val = sess.run(model.prediction(), feed_dict={x: np.expand_dims(val_x, axis=-1), y: val_y})

    # print(np.all(pred_val_saved == pred_val))

    num_of_inputs = len(sys.argv)
    with open('../Datasets/BaseDataset/dota2_dataset_preprocessed.pickle', 'rb') as file:
        data = pk.load(file)
    if num_of_inputs != 3:
        pass

        dataSize = data.shape[0]
        teamsDict = {}
        for i in range(dataSize):
            teamsDict[data.at[data.index[i], 'dire_team']['name']] = int(data.at[data.index[i], 'dire_team_id'])

        with open('../temp.json', 'w') as temp_file:
            json.dump(teamsDict, temp_file)
        print("1")
    else:
        team_dire_id = int(sys.argv[1], base=10)
        team_radiant_id = int(sys.argv[2], base=10)
        dire_matches = data.query('dire_team_id == @team_dire_id | radiant_team_id == @team_dire_id')
        last_match_dire = dire_matches.loc[dire_matches.index[-1]]
        if last_match_dire['dire_team_id'] == team_dire_id:
            avrgDireScore = last_match_dire['avg_dire_score']
        else:
            avrgDireScore = last_match_dire['avg_radiant_score']

        radiant_matches = data.query('dire_team_id == @team_radiant_id | radiant_team_id == @team_radiant_id')
        last_match_dire = dire_matches.loc[dire_matches.index[-1]]
        if last_match_dire['dire_team_id'] == team_radiant_id:
            avrgRadiantScore = last_match_dire['avg_dire_score']
        else:
            avrgRadiantScore = last_match_dire['avg_radiant_score']

        match_patch = data.at[data.index[-1], 'patch']
        match_avg_dur = data.at[data.index[-1], 'avg_duration']

        match_region = 5
        to_analize = np.asarray([[avrgDireScore, avrgRadiantScore, match_avg_dur, match_patch, match_region,
                team_dire_id, team_radiant_id]])

        with tf.Session() as sess:
            saver.restore(sess, "model.ckpt")
            pred_val = sess.run(model.prediction(), feed_dict={x: np.expand_dims(to_analize, axis=-1)})
        print(str(pred_val[0][0]))
