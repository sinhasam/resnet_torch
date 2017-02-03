require 'nn'
require 'torch'
-- require 'cudnn'
-- require 'cunn'

local SpatialConvolution = nn.SpatialConvolution
local AvgPool = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local BatchNorm = nn.SpatialBatchNormalization
local MaxPool = nn.SpatialMaxPooling


local function firstBlock()
	local model = nn.Sequential()
	model:add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3)) -- 3 by 3 pad
	model:add(BatchNorm(64))
	model:add(ReLU())
	model:add(MaxPool(3, 3, 2, 2))
end


local function basicBlock(dim)
	local new = nn.Sequential()
	new:add(SpatialConvolution(3, dim, 3, 3, 1, 1)) -- no pad
	new:add(BatchNorm(dim))
	new:add(ReLU())
	new:add(SpatialConvolution(dim, dim, 3, 3, 1, 1))
	new:add(BatchNorm(dim))
	
	local shortCut = nn.Sequential()
	shortCut:add(nn.ConcatTable():add(new):add(nn.Identity())) -- need to test if this way of adding residue works
	shortCut:add(nn.CAddTable())
	shortCut:add(ReLU)
	return shortCut
end

local model = nn.Sequential()
model:add(firstBlock())

for i=1, 4 do
	model:add(basicBlock(64))
end

-- model:
