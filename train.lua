require 'nn'
require 'torch'
require 'optim'
residual = 'residual'
require'data'


opt = {
	batchSize = 1, 
	learningRate = 0.0001, 
	numEpoch = 10000
}


model = nn.Sequential()

model:add(makeModel())

local totalDataSize = getNumDataSize()

