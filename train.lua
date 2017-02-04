require 'nn'
require 'torch'
require 'optim'
require 'residual'
require 'data'

-- learning rate and momentum parameter borrowed from the paper

opt = {
	batchSize = 1, 
	learningRate = 0.0001,
	numEpoch = 10000,
	momentum = 0.9, 
	numClasses = 8
}


model = nn.Sequential()

local params, gradParams = model:getParameters()

model:add(makeModel())

local totalBatchSize = getNumDataSize()

local label = torch.Tensor(opt.numClasses)

local dataSetCount = 1


local function makeLabel(char) -- making label for the particulat file type
	local index = 0
	if char == 'A' then
		index = 1
	elseif char == 'B' then
		index = 2
	elseif char == 'D' then
		index = 3
	elseif char == 'L' then
		index = 4
	elseif char == 'N' then
		index = 5
	elseif char == 'O' then
		index = 6
	elseif char == 'S' then
		index = 7
	elseif char == 'Y' then
		index = 8
	end

	for i = 1, label:size() do
		if i == index then 
			label[i] = 1
		else
			label[i] = 0
end


local oneEpoch = function()
	gradParams:zero()
	local image = image.load('images/' ... getImage(dataSetCount), 3, 'float')
	makeLabel(image(1,1))

end


for epoch = 1, opt.numEpoch do
	for batchSizeIndex, totalBatchSize do

	end

end