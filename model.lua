require 'nn'
require 'paths'
require 'torch'
require 'image'
-- require 'cudnn'
-- require 'cunn'
require 'optim'


options = {
    batchSize = nil
    learningRate = 0.0002
    numEpoch = 100000
    gpu = 0
}

local Convolution = nn.SpatialConvolution
local AveragePool = nn.SpatialAveragePooling
local Relu = nn.ReLU    
local MaxPool = nn.SpatialMaxPooling

