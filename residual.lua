require 'nn'
require 'torch'
-- require 'cudnn'
-- require 'cunn'


local SpatialConvolution = nn.SpatialConvolution
local AvgPool = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local BatchNorm = nn.SpatialBatchNormalization
local MaxPool = nn.SpatialMaxPooling


local numClasses = 8


local function firstBlock()
	local model = nn.Sequential()
	model:add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3)) -- 3 by 3 pad
	model:add(BatchNorm(64))
	model:add(ReLU(true))
	model:add(MaxPool(3, 3, 2, 2))
	return model
end


local function basicBlock(inputDim, outputDim, stride)
	local new = nn.Sequential()
	new:add(SpatialConvolution(inputDim, outputDim, 3, 3, stride, stride, 1, 1)) -- no pad
	new:add(BatchNorm(outputDim))
	new:add(ReLU(true))
	new:add(SpatialConvolution(outputDim, outputDim, 3, 3, 1, 1, 1, 1))
	new:add(BatchNorm(outputDim))
	
	local shortCut = nn.Sequential()
	shortCut:add(nn.ConcatTable():add(new):add(nn.Identity())) -- need to test if this way of adding residue works
	shortCut:add(nn.CAddTable())
	shortCut:add(ReLU(true))
	return shortCut
end


function makeModel() -- 18 layer model
	model = nn.Sequential()

	model:add(firstBlock()) -- 224*224 -> 112*112
	model:add(basicBlock(3, 64, 1))
	
	model:add(basicBlock(64, 64, 1))
	model:add(basicBlock(64, 128, 2)) -- 56*56

	model:add(basicBlock(128, 128, 1))
	model:add(basicBlock(128, 256, 2)) -- 28*28

	model:add(basicBlock(256, 256, 1))
	model:add(basicBlock(256, 512, 2)) -- 14*14

	model:add(basicBlock(512, 512, 1))
	model:add(basicBlock(512, 512, 2)) -- 7*7

	model:add(AvgPool(7, 7, 1, 1))
	model:add(nn.View(512))
	model:add(nn.Linear(512, 120))
	model:add(ReLU(true))
	model:add(nn.Linear(120, 84))
	model:add(nn.ReLU(true))
	model:add(nn.Linear(84, 10)) -- 10 output classes
	model:add(nn.LogSoftMax())
	return model
end

return model
