import pickle as pk
import numpy as np


class PlayerBasedDataReader:

    def __init__(self, filename: str):
        self.raw_data = None
        self.valide_data = None
        self.read_dataset(filename)
        self.preprocess_data()

    def read_dataset(self, filename: str):
        with open(filename, 'rb') as file:
            self.raw_data = pk.load(file)

    def preprocess_data(self):
        new_data = self.raw_data.query('dire_score !=0 & radiant_score!=0')
        new_data = new_data.query('duration >899')
        new_data = new_data.sort_values(by=['start_time'])
        data_size = new_data.shape[0]
        for i in range(1, 6):
            new_data['dire_player_' + str(i)] = np.zeros(data_size)
            new_data['radiant_player_' + str(i)] = np.zeros(data_size)
        for i in range(data_size):
            for j in range(5):
                new_data.at[i, 'radiant_player_'+str(j+1)] = new_data.iloc[i]['players_radiant_id'][j]
                new_data.at[i, 'dire_player_'+str(j+1)] = new_data.iloc[i]['players_dire_id'][j]
        new_data = new_data.drop(columns=['players_radiant_id', 'players_dire_id'])
        self.valide_data = new_data

    def write_data(self, filename: str):
        with open(filename, 'wb') as file:
            pk.dump(self.valide_data, file)


if __name__ == '__main__':
    data = PlayerBasedDataReader('../BaseDataset/dota2_dataset.pickle')
    data.write_data('PlayerBasedData.pickle')
    print(data.valide_data['radiant_player_1'])
