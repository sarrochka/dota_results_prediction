import pickle as pk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataReader:

    def __init__(self, filename: str, feature_cols: list, y_cols, x_cols):
        self.raw_data = None
        self.preprocessed_data = None
        self.feature_cols = feature_cols
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.used_cols = feature_cols + y_cols

        self.read_dataset(filename)

    def read_dataset(self, filename: str):
        with open(filename, 'rb') as file:
            self.raw_data = pk.load(file)

    def read_preprocessed(self, filename: str):
        with open(filename, 'rb') as file:
            self.preprocessed_data = pk.load(file)

    def preprocess_data(self):
        new_data = self.raw_data.query('dire_score !=0 & radiant_score!=0')
        new_data = new_data.query('duration >899')

        new_data = new_data.sort_values(by=['start_time'])
        new_data = new_data[self.used_cols].dropna()

        data_size = new_data.shape[0]
        for i in range(1, 6):
            new_data[f'dire_player_{i}'] = np.zeros(data_size, dtype=np.int64)
            new_data[f'radiant_player_{i}'] = np.zeros(data_size, dtype=np.int64)

        idx_r = new_data.columns.get_loc('players_radiant_id')
        idx_d = new_data.columns.get_loc('players_dire_id')

        players_radiant_id = np.asarray(new_data.iloc[:, idx_r].values.tolist())
        players_dire_id = np.asarray(new_data.iloc[:, idx_d].values.tolist())

        for j in range(5):
            new_data.loc[:, f'radiant_player_{j + 1}'] = players_radiant_id[:, j]
            new_data.loc[:, f'dire_player_{j + 1}'] = players_dire_id[:, j]

        new_data = new_data.drop(columns=['players_radiant_id', 'players_dire_id'])

        new_data = new_data.dropna()

        # add average kills and game time for each game except the first patch as feature
        new_data['avg_dire_score'] = np.zeros(data_size)
        new_data['avg_radiant_score'] = np.zeros(data_size)
        new_data['avg_duration'] = np.zeros(data_size)

        patch_avg = {}
        patches, idxs = np.unique(new_data.loc[:, 'patch'], return_index=True)

        for patch in patches:
            patch_avg[patch] = np.mean(new_data.loc[:, ['dire_score', 'radiant_score', 'duration']].
                                       loc[new_data['patch'] == patch], axis=0)

        # don't change avg stats from real for first patch
        first_patch_rows = new_data['patch'] == patches[0]
        first_patch_data = new_data.loc[first_patch_rows]

        new_data.loc[first_patch_rows, 'avg_radiant_score'] = first_patch_data['radiant_score']
        new_data.loc[first_patch_rows, 'avg_dire_score'] = first_patch_data['dire_score']
        new_data.loc[first_patch_rows, 'avg_duration'] = first_patch_data['duration']

        start = idxs[1]

        for i in new_data.index[start:]:
            match = new_data.loc[i]
            patch = match['patch']
            radiant_team_id = match['radiant_team_id']
            dire_team_id = match['dire_team_id']

            prev_games_this_patch = new_data.iloc[:i].loc[new_data['patch'] == patch]

            radiant_team_prev_matches_this_patch_as_radiant = prev_games_this_patch.loc[
                prev_games_this_patch['radiant_team_id'] == radiant_team_id]
            radiant_team_prev_matches_this_patch_as_dire = prev_games_this_patch.loc[
                new_data['dire_team_id'] == radiant_team_id]
            dire_team_prev_matches_this_patch_as_radiant = prev_games_this_patch.loc[
                prev_games_this_patch['radiant_team_id'] == dire_team_id]
            dire_team_prev_matches_this_patch_as_dire = prev_games_this_patch.loc[
                new_data['dire_team_id'] == dire_team_id]

            duration = np.empty(dtype=np.float32, shape=(0,))

            if radiant_team_prev_matches_this_patch_as_radiant.shape[0] != 0:
                new_data.at[i, 'avg_radiant_score'] = np.mean(radiant_team_prev_matches_this_patch_as_radiant
                                                              ['radiant_score'])
                duration = np.append(duration, radiant_team_prev_matches_this_patch_as_radiant['duration'])
            elif radiant_team_prev_matches_this_patch_as_dire.shape[0] != 0:
                new_data.at[i, 'avg_radiant_score'] = np.mean(
                    radiant_team_prev_matches_this_patch_as_dire['dire_score'])
                duration = np.append(duration, radiant_team_prev_matches_this_patch_as_dire['duration'])
            elif prev_games_this_patch.shape[0] != 0:
                new_data.at[i, 'avg_radiant_score'] = np.mean(prev_games_this_patch['radiant_score'])
                duration = np.append(duration, prev_games_this_patch['duration'])
            else:
                new_data.at[i, 'avg_radiant_score'] = patch_avg[patch - 1][1]
                duration = np.append(duration, patch_avg[patch - 1][2])

            if dire_team_prev_matches_this_patch_as_dire.shape[0] != 0:
                new_data.at[i, 'avg_dire_score'] = np.mean(dire_team_prev_matches_this_patch_as_dire['dire_score'])
                duration = np.append(duration, dire_team_prev_matches_this_patch_as_dire['duration'])
            elif dire_team_prev_matches_this_patch_as_radiant.shape[0] != 0:
                new_data.at[i, 'avg_dire_score'] = np.mean(
                    dire_team_prev_matches_this_patch_as_radiant['radiant_score'])
                duration = np.append(duration, dire_team_prev_matches_this_patch_as_radiant['duration'])
            elif prev_games_this_patch.shape[0] != 0:
                new_data.at[i, 'avg_dire_score'] = np.mean(prev_games_this_patch['dire_score'])
                duration = np.append(duration, prev_games_this_patch['duration'])
            else:
                new_data.at[i, 'avg_dire_score'] = patch_avg[patch - 1][0]
                duration = np.append(duration, patch_avg[patch - 1][2])

            new_data.at[i, 'avg_duration'] = np.mean(duration)

            if new_data.at[i, 'avg_duration'] == 0 or new_data.at[i, 'avg_dire_score'] == 0:
                print(i)
                print(prev_games_this_patch.shape, radiant_team_prev_matches_this_patch_as_radiant.shape,
                      radiant_team_prev_matches_this_patch_as_dire.shape)
                raise ValueError(i)

        self.preprocessed_data = new_data

    def write_data(self, filename: str):
        with open(filename, 'wb') as file:
            pk.dump(self.preprocessed_data, file)

    def get_x(self):
        return self.preprocessed_data[self.x_cols]

    def get_y(self):
        return pd.get_dummies(self.preprocessed_data[self.y_cols])

    def get_train_val_test(self, train_size=0.7):
        all_x = self.get_x()
        all_y = self.get_y()
        train_x, val_test_x, train_y, val_test_y = train_test_split(all_x, all_y, test_size=1. - train_size)
        val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5)

        return train_x, train_y, val_x, val_y, test_x, test_y


if __name__ == '__main__':
    feature_cols_ = ['dire_score', 'radiant_score', 'duration', 'patch', 'region', 'radiant_team_id', 'dire_team_id',
                     'players_radiant_id', 'players_dire_id', 'radiant_team', 'dire_team']
    x_cols_ = ['avg_dire_score', 'avg_radiant_score', 'avg_duration', 'patch', 'region', 'radiant_team_id',
               'dire_team_id'] + [f'radiant_player_{j}' for j in range(1, 6)] + \
              [f'dire_player_{j}' for j in range(1, 6)]
    y_cols_ = ['radiant_win']
    data = DataReader('../BaseDataset/dota2_dataset.pickle', feature_cols_, y_cols_, x_cols_)
    data.preprocess_data()
    data.write_data('../BaseDataset/dota2_dataset_preprocessed.pickle')
    print(data.get_x())
