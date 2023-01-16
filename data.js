const config = require("./config.js");
const { datasetUrl, frameWidth, frameHeight, spriteCount, imageWidth, imageHeight, imageChannels, testInstances } = config;

const reorderDataset = async (dataset, columns) => {
	// Split the spritesheet into individual image and label image tensors
	const inputs = [];
	const targets = [];

	for (let i = 0; i < spriteCount; i++) {
		const x = i % columns;
		const y = (i - x) / columns;

		const input = dataset.slice([x, y, 0], [frameWidth, frameHeight, imageChannels]);
  	const target = dataset.slice([x + frameWidth, y, 0], [frameWidth, frameHeight, imageChannels]);
		
		const paddedInput = input.pad([[0, imageWidth - frameWidth], [0, imageHeight - frameHeight], [0, 0]], 0);
		const paddedTarget = target.pad([[0, imageWidth - frameWidth], [0, imageHeight - frameHeight], [0, 0]], 0);

  	inputs.push(paddedInput.expandDims());
  	targets.push(paddedTarget.expandDims());
	}
		
	// Concatenate the input and target tensors into a single tensor
	const data = tf.concat([tf.stack(inputs, 1), tf.stack(targets, 1)]);
	
	return data;
};

const splitData = (data) => {
	return tf.split(data, [spriteCount - testInstances, testInstances], 1);
};

export const loadData = async () => {
  // Load the spritesheet and convert to a tensor
	const spritesheet = await loadImage(datasetUrl);
	const dataset = await imageToTensor(spritesheet, imageChannels);

	// Reorder the dataset so that there are collections of input and target inputs with matching indices
	const columns = spritesheet.width / (frameWidth * 2);
	const data = await reorderDataset(dataset, columns);

	// Split the data into a training set and a test set
	const [trainData, testData] = splitData(data);

	// Split training and testing sets into source and target sets
	const [trainX, trainY] = tf.unstack(trainData);
	const [testX, testY] = tf.unstack(testData);
	
	return { trainX, trainY, testX, testY };
};
