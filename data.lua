require 'lfs'


images = {}
datasetCount = 1

for image in lfs.dir('images') do
	if (string.find(image, '.jpg')) then
		images[datasetCount] = image
		datasetCount = datasetCount + 1
	end
end	

function getNumDataSize()
	return datasetCount
end

function getImage(index)
	return images[index]
end