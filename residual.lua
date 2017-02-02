require 'nn'
require 'torch'
-- require 'cudnn'
-- require 'cunn'


local SpatialConvolution = nn.SpatialConvolution
local AvgPool = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local BatchNorm = nn.SpatialBatchNormalization

model = nn.Sequential()
opt = {
	inputDim = 64	
}

local function firstBlock()
	model:add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3)) -- 3 by 3 pad
	model:add(BatchNorm(64))
	model:add(ReLU())
	model:add(nn.AvgPool(3, 3, 2, 2))
end


local function basicBlock()
	new = nn.Sequential()
	new:add(SpatialConvolution(opt.inputDim, opt.inputDim, 3, 3, 1, 1)) -- no pad
	new:add(BatchNorm(opt.inputDim))
	new:add(ReLU())
	new:add(SpatialConvolution(opt.inputDim, opt.inputDim, 3, 3, 1, 1))
	new:add(BatchNorm(opt.inputDim))
	
	local shortCut = nn.Sequential()
	shortCut:add(nn.ConcatTable():add(new):add(nn.Identity())) -- need to test if this way of adding residue works
	shortCut:add(nn.CAddTable())
	shortCut:add(ReLU)
	return shortCut
end
