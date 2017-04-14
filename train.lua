require 'nn'
require 'torch'
require 'optim'
require 'residual'
require 'data'
require 'image'
-- learning rate and momentum parameter borrowed from the paper

opt = {
	batchSize = 1, 
	learningRate = 0.0001,
	numEpoch = 500,
	momentum = 0.9, 
	numClasses = 8,
}

local criterion = nn.BCECriterion()
local model = nn.Sequential()
local imageInput = torch.Tensor(opt.batchSize, 3, 224, 224)

local modelParams, gradModelParams = model:getParameters()

model:add(makeModel())

local totalBatchSize = getNumDataSize()

local label = torch.Tensor(opt.batchSize)

local dataSetCount = 1

local params, gradParams = model:getParameters()

local function makeLabel(firstLetter) -- making label for the particulat file type
	if firstLetter == 'A' then
		label:fill(0)
	elseif firstLetter == 'B' then
		label:fill(1)
	elseif firstLetter == 'D' then
		label:fill(2)
	elseif firstLetter == 'L' then
		label:fill(3)
	elseif firstLetter == 'N' then
		label:fill(4)
	elseif firstLetter == 'O' then
		label:fill(5)
	elseif firstLetter == 'S' then
		label:fill(6)
	elseif firstLetter == 'Y' then
		label:fill(7)
	end
end

optimState = {
	learningRate = opt.learningRate, 
	momentum = opt.momentum, -- as used in the paper
}

local oneEpoch = function(x)
	gradParams:zero()
	local imageName = getImage(dataSetCount)
	dataSetCount = dataSetCount + 1
	local img = image.load(('images/'..imageName), 3, 'float')
	makeLabel(imageName:sub(1,1))
	
	imageInput:copy(img)

	local output = model:forward(imageInput)
	local imgError = criterion:forward(output, label)
	local criterionError = criterion:backward(output, label)

	model:backward(imageInput, criterionError)

	return imgError, gradParams
end


for epoch = 1, opt.numEpoch do
	print('epoch count: ' .. epoch)
	for batchSizeIndex = 1, totalBatchSize do
		optim.adam(oneEpoch, modelParams, optimState)

		if batchSizeIndex % 200 == 0 then 
			print('batch size count '.. batchSizeIndex)
		end
	end
	modelParams, gradModelParams = nil, nil

	if epoch % 50 == 0 then
		torch.save('TrainedModels/'..epoch, model:clearState()) -- for memory
	end

	modelParams, gradModelParams = model:getParameters()	
	dataSetCount = 1
end
