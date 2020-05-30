import pickle as pk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class PlayerBasedDataReader:

    def __init__(self, filename: str, feature_cols: list, y_cols):
        self.raw_data = None
        self.preprocessed_data = None
        self.feature_cols = feature_cols
        self.y_cols = y_cols
        self.used_cols = feature_cols + y_cols

        self.read_dataset(filename)
        self.preprocess_data()

    def read_dataset(self, filename: str):
        with open(filename, 'rb') as file:
            self.raw_data = pk.load(file)

    def preprocess_data(self):
        new_data = self.raw_data.query('dire_score !=0 & radiant_score!=0')
        new_data = new_data.query('duration >899')

        new_data = new_data.sort_values(by=['start_time'])
        new_data = new_data[self.used_cols].dropna()

        data_size = new_data.shape[0]
        for i in range(1, 6):
            new_data[f'dire_player_{i}'] = np.zeros(data_size)
            new_data[f'radiant_player_{i}'] = np.zeros(data_size)

        idx_r = new_data.columns.get_loc('players_radiant_id')
        idx_d = new_data.columns.get_loc('players_dire_id')

        players_radiant_id = np.asarray(new_data.iloc[:, idx_r].values.tolist())
        players_dire_id = np.asarray(new_data.iloc[:, idx_d].values.tolist())

        for j in range(5):
            new_data.loc[:, f'radiant_player_{j + 1}'] = players_radiant_id[:, j]
            new_data.loc[:, f'dire_player_{j + 1}'] = players_dire_id[:, j]

        new_data = new_data.drop(columns=['players_radiant_id', 'players_dire_id'])
        self.feature_cols.remove('players_radiant_id')
        self.feature_cols.remove('players_dire_id')

        self.preprocessed_data = new_data

    def swap_players_column(self, data_frame, player_is_dire):
        cols = data_frame.columns.tolist()
        if player_is_dire:
            cols = cols[:11]
            for j in range(1, 6):
                cols.append('dire_player_'+str(j))
            for j in range(1, 6):
                cols.append('radiant_player_'+str(j))
        else:
            cols = cols[:11]
            for j in range(1, 6):
                cols.append('radiant_player_'+str(j))
            for j in range(1, 6):
                cols.append('dire_player_'+str(j))
        data_frame = data_frame[cols]
        return data_frame

    def get_data_by_player(self, player_id: int):
        data_frame = self.preprocessed_data.query('dire_player_1 == @player_id | dire_player_2 == @player_id | '
                                                  'dire_player_3 == @player_id | dire_player_4 == @player_id | '
                                                  'dire_player_5 == @player_id | radiant_player_1 == @player_id | '
                                                  'radiant_player_2 == @player_id | radiant_player_3 == @player_id |'
                                                  'radiant_player_4 == @player_id | radiant_player_5 == @player_id')

        data_size = data_frame.shape[0]
        for i in range(data_size):
            for j in range(1, 6):
                if player_id == data_frame.iloc[i]['dire_player_'+str(j)]:
                    data_frame.at[data_frame.index[i], 'dire_player_' + str(j)] = data_frame.iloc[i]['dire_player_1']
                    data_frame.at[data_frame.index[i], 'dire_player_1'] = player_id
                elif player_id == data_frame.iloc[i]['radiant_player_'+str(j)]:
                    data_frame.at[data_frame.index[i], 'radiant_player_' + str(j)] = data_frame.iloc[i]['radiant_player_1']
                    data_frame.at[data_frame.index[i], 'radiant_player_1'] = player_id

        return data_frame

    def write_data(self, filename: str):
        with open(filename, 'wb') as file:
            pk.dump(self.preprocessed_data, file)

    def find_unique_players(self):
        pass

    def get_x(self):
        return self.preprocessed_data[self.feature_cols]

    def get_y(self):
        return pd.get_dummies(self.preprocessed_data[self.y_cols])

    def get_train_val_test(self, train_size=0.7):
        all_x = self.get_x()
        all_y = self.get_y()
        train_x, val_test_x, train_y, val_test_y = train_test_split(all_x, all_y, test_size=1.-train_size)
        val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5)

        return train_x, train_y, val_x, val_y, test_x, test_y


if __name__ == '__main__':
    feature_cols_ = ['dire_score', 'radiant_score', 'duration', 'patch', 'region', 'radiant_team_id', 'dire_team_id',
                     'players_radiant_id', 'players_dire_id']
    y_cols_ = ['radiant_win']
    data = PlayerBasedDataReader('../BaseDataset/dota2_dataset.pickle', feature_cols_, y_cols_)
    print(data.get_data_by_player(117956848))
    # data.write_data('PlayerBasedData.pickle')
    # print(data.valide_data['radiant_player_1'])
