import { datasetUrl, frameWidth, frameHeight, spriteCount, imageWidth, imageHeight, imageChannels, testInstances } from "./config.js";

export const tensorToImage = async (tensor) => {
	const canvas = new OffscreenCanvas(tensor.shape[0], tensor.shape[1]);
	await tf.browser.toPixels(tensor, canvas);
	const image = new Image();
	image.src = canvas.transferToImageBitmap();
	return image;
};

export const loadDataset = async () => {
  // Load the spritesheet into a tensor
	const spritesheet = new Image();
	spritesheet.src = datasetUrl;
	const dataset = await tf.browser.fromPixelsAsync(spritesheet, imageChannels);

	// Split the spritesheet into individual image and label image tensors
	const inputImages = [];
	const targetImages = [];
	const columns = spritesheet.width / (frameWidth * 2);

	for (let i = 0; i < spriteCount; i++) {
		const x = i % columns;
		const y = (i - x) / columns;

		const inputImage = dataset.slice([x, y, 0], [frameWidth, frameHeight, imageChannels]);
  	const targetImage = dataset.slice([x + frameWidth, y, 0], [frameWidth, frameHeight, imageChannels]);
		
		const paddedInputImage = inputImage.pad([[0, imageWidth - frameWidth], [0, imageHeight - frameHeight], [0, 0]], 0);
		const paddedTargetImage = targetImage.pad([[0, imageWidth - frameWidth], [0, imageHeight - frameHeight], [0, 0]], 0);

  	inputImages.push(paddedInputImage.expandDims());
  	targetImages.push(paddedTargetImage.expandDims());
	}

	// Concatenate the image and label image tensors into a single tensor
	const data = tf.concat([tf.stack(inputImages, 1), tf.stack(targetImages, 1)]);

	// Split the data into a training set and a test set
	const [trainData, testData] = tf.split(data, [spriteCount - testInstances, testInstances], 1);

	// TODO: Shuffle the data

	return { trainData, testData };
};
