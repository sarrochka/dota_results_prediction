import pickle as pk


class DataReader:

    def __init__(self, filename: str):
        self.raw_data = None
        self.valide_data = None
        self.read_dataset(filename)
        self.validate_data()

    def read_dataset(self, filename: str):
        with open(filename, 'rb') as file:
            self.raw_data = pk.load(file)

    def validate_data(self):
        new_data = self.raw_data.query('dire_score !=0 & radiant_score!=0')
        new_data = new_data.query('duration >899')
        self.valide_data = new_data

    def write_data(self, filename: str):
        with open(filename, 'wb') as file:
            pk.dump(self.valide_data, file)


if __name__ == '__main__':
    reader = DataReader('dota2_dataset.pickle')
    reader.write_data('validated_data.pickle')
    print(reader.valide_data)
