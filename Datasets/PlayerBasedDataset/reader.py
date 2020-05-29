import pickle as pk
import numpy as np


class PlayerBasedDataReader:

    def __init__(self, filename: str, feature_cols: list, y_cols):
        self.raw_data = None
        self.valide_data = None
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
        print(players_radiant_id.shape, players_radiant_id.dtype)

        for j in range(5):
            new_data.loc[:, f'radiant_player_{j + 1}'] = players_radiant_id[:, j]
            new_data.loc[:, f'dire_player_{j + 1}'] = players_dire_id[:, j]

        new_data = new_data.drop(columns=['players_radiant_id', 'players_dire_id'])

        self.valide_data = new_data

    def write_data(self, filename: str):
        with open(filename, 'wb') as file:
            pk.dump(self.valide_data, file)

    def find_unique_players(self):
        pass


if __name__ == '__main__':
    feature_cols_ = ['dire_score', 'radiant_score', 'duration', 'patch', 'region', 'radiant_team_id', 'dire_team_id',
                     'players_radiant_id', 'players_dire_id']
    y_cols_ = ['radiant_win']
    data = PlayerBasedDataReader('../BaseDataset/dota2_dataset.pickle', feature_cols_, y_cols_)
    # data.write_data('PlayerBasedData.pickle')
    # print(data.valide_data['radiant_player_1'])
