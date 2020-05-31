import sys

import tensorflow as tf
import numpy as np
import pickle as pk
import pandas as pd
import os
import json


os.chdir('../model')

from reader import DataReader
from nn import ResultPredictingModel



def build_model(**kwargs):
    n_x = 7
    n_y = 2

    x = tf.placeholder(tf.float32, shape=[None, n_x, 1], name="x")
    y = tf.placeholder(tf.float32, shape=[None, n_y], name="y")

    h = [500, 100, 50]
    model = ResultPredictingModel(x_=x, y_=y, h_list=h, **kwargs)
    model.initialize()

    return model, x, y


def predict(input_to_analyze: np.ndarray):
    model, x, y = build_model(cost_weigths_=np.asarray([0.5, 0.5]))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "model.ckpt")
        predicted = sess.run(model.prediction(), feed_dict={x: np.expand_dims(input_to_analyze, axis=-1)})

    return predicted


def update(input_to_update: pd.DataFrame):
    feature_cols = ['dire_score', 'radiant_score', 'duration', 'patch', 'region', 'radiant_team_id', 'dire_team_id']
    y_cols = ['radiant_win']
    x_cols = ['avg_dire_score', 'avg_radiant_score', 'avg_duration', 'patch', 'region']
    x_cols += ['radiant_team_id', 'dire_team_id']
    #x_cols += [f'radiant_player_{j}' for j in range(1, 6)] + [f'dire_player_{j}' for j in range(1, 6)]

    data_reader = DataReader('../Datasets/BaseDataset/dota2_dataset.pickle', feature_cols, y_cols, x_cols)
    data_reader.read_preprocessed('../Datasets/BaseDataset/dota2_dataset_preprocessed.pickle')
    input_to_update = data_reader.add_observations(input_to_update)
    data_reader.write_data('../Datasets/BaseDataset/dota2_dataset_preprocessed.pickle')

    radiant_wr = np.where(data_reader.preprocessed_data[y_cols])[0].shape[0] / \
                 data_reader.preprocessed_data[y_cols].shape[0]

    cost_weigths = np.asarray([radiant_wr, 1. - radiant_wr])
    lr = 1e-5

    model, x, y = build_model(cost_weigths_=cost_weigths, learning_rate=lr)

    train_x = np.expand_dims(input_to_update[x_cols], axis=-1)
    train_y = np.hstack((input_to_update[y_cols], 1-input_to_update[y_cols]))

    print(train_y)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "model.ckpt")
        _, c = sess.run([model.optimize(), model.cost()], feed_dict={x: train_x,
                                                                     y: train_y})
        saver.save(sess, "model.ckpt")


if __name__ == '__main__':
    num_of_inputs = len(sys.argv)
    with open('../Datasets/BaseDataset/dota2_dataset_preprocessed.pickle', 'rb') as file:
        data = pk.load(file)
    if num_of_inputs == 9:
        team_d_id = int(sys.argv[1])
        team_r_id = int(sys.argv[2])
        match_id = int(sys.argv[3])
        duration = int(sys.argv[4])
        region_id = int(sys.argv[5])
        patch_id = int(sys.argv[6])
        team_d_score = int(sys.argv[7])
        team_r_score = int(sys.argv[8])

        temp_dict = {}
        temp_dict['dire_score'] = team_d_score
        temp_dict['radiant_score'] = team_r_score
        temp_dict['duration'] = duration
        temp_dict['patch'] = patch_id
        temp_dict['region'] = region_id
        temp_dict['radiant_team_id'] = team_r_id
        temp_dict['dire_team_id'] = team_d_id
        temp_dict['dire_team_id'] = team_d_id

        temp_df = pd.DataFrame.from_dict(temp_dict)
        update(temp_df)
        print("1")

    elif num_of_inputs != 3:
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
        to_analyze = np.asarray([[avrgDireScore, avrgRadiantScore, match_avg_dur, match_patch, match_region,
                team_dire_id, team_radiant_id]])

        pred_val = predict(to_analyze)
        print(str(pred_val[0][0]))
