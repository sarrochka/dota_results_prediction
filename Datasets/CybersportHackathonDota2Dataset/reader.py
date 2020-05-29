import pickle as pk
import numpy as np

file = open("validated_data.pickle","rb")
data = pk.load(file)
file.close()

print (data)