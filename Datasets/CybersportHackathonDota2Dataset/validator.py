import pickle as pk
import numpy as np

file = open("dota2_dataset.pickle","rb")
data = pk.load(file)
file.close()

newDat = data.query('dire_score !=0 & radiant_score!=0')
newDat = newDat.query('duration >899')
pk.dump(newDat, open("validated_data.pickle","wb"))
print("success")