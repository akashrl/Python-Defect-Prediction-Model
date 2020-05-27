import pickle
import pandas as pd
import numpy as np

with open('data/y_train.pickle', 'rb') as handle:
	Y_train = pickle.load(handle)

Y_train = Y_train[:50000]

for i in [[Y_train]]:
	print(i)

TrCount = 0
TrCount = (Y_train.count(1))

print('This is 1 count in Y_train: ', TrCount)

buggyrate_y_train = TrCount/50000

print('This is buggyrate of Y_Train: ', buggyrate_y_train)
print("\n")


with open('data/y_test.pickle', 'rb') as handle:
        Y_test = pickle.load(handle)

Y_test = Y_test[:25000]

TsCount = 0
TsCount = (Y_test.count(1))

print('This is 1 count in Y_test: ', TsCount)

buggyrate_y_test = TsCount/25000

print('This is buggyrate of Y_test: ', buggyrate_y_test)
print("\n")

with open('data/y_valid.pickle', 'rb') as handle:
        Y_valid = pickle.load(handle)

Y_valid = Y_valid[:25000]

VaCount = 0 
VaCount = (Y_valid.count(1))

print('This is 1 count in Y_valid: ', VaCount)

buggyrate_y_valid = VaCount/25000

print('This is buggyrate of Y_valid: ', buggyrate_y_valid)
