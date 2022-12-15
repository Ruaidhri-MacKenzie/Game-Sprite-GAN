import { fileName, frameWidth, frameHeight, spriteCount, inputShape, batchSize } from "./config.js";

const imageToTensor = async (image) => {
	const bitmap = await createImageBitmap(image);
	const tensor = tf.browser.fromPixels(bitmap);
	return tensor;
};

const tensorToImage = async (tensor) => {
	const canvas = new OffscreenCanvas(tensor.shape.width, tensor.shape.height);
	await tf.browser.toPixels(tensor, canvas);
	const image = new Image();
	image.src = canvas.toDataURL("image/png");
	return image;
};

const cropImageToTensor = (image, x, y, width, height) => {
	const canvas = new OffscreenCanvas(image.width, image.height);
	const ctx = canvas.getContext("2d");
	ctx.drawImage(image, 0, 0);

	const imageData = ctx.getImageData(x, y, width, height);
	const tensor = tf.browser.fromPixels(imageData);
	return tensor;
};

export const loadDataset = () => {
	// load spritesheet
	const spritesheet = new Image();
	spritesheet.src = `/data/${fileName}`;
	const columns = spritesheet.width / (frameWidth * 2);

	// create dataset of { input, target } instances
	const dataset = [];

	for (let i = 0; i < spriteCount; i++) {
		const x = i % columns;
		const y = (i - x) / columns;
		
		const input = cropImageToTensor(spritesheet, x, y, frameWidth, frameHeight);
		const target = cropImageToTensor(spritesheet, x + frameWidth, y, frameWidth, frameHeight);

		dataset.push({ input, target });
	}

	return dataset;
};

export const getInputs = (dataset) => dataset.map(instance => instance.input);
export const getTargets = (dataset) => dataset.map(instance => instance.target);
export const generateNoise = () => tf.randomNormal([batchSize, inputShape[0], inputShape[1], inputShape[2]]);
