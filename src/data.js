import { fileName, frameWidth, frameHeight, spriteCount, inputShape, batchSize } from "./config.js";

const cropImage = (image, x, y, width, height) => {
	const canvas = document.createElement("canvas");
	canvas.width = image.width;
	canvas.height = image.height;
	
	const ctx = canvas.getContext("2d");
	ctx.drawImage(image, 0, 0);
	ctx.drawImage(canvas, x, y, width, height, 0, 0, width, height);
	
	const croppedImage = new Image();
	croppedImage.src = canvas.toDataURL("image/png");
	return croppedImage;
};

const imageToTensor = async (image) => {
	const bitmap = await createImageBitmap(image);
	const tensor = tf.browser.fromPixels(bitmap);
	return tensor;
};

const tensorToImage = async (tensor) => {
	const canvas = document.createElement("canvas");
	canvas.width = tensor.shape.width;
	canvas.height = tensor.shape.height;
	await tf.browser.toPixels(tensor, canvas);
	const image = new Image();
	image.src = canvas.toDataURL("image/png");
	return image;
};

export const loadDataset = async () => {
	// load spritesheet
	const spritesheet = new Image();
	spritesheet.src = `/data/${fileName}`;
	const columns = spritesheet.width / (frameWidth * 2);

	// split into { input, target } instances
	const dataset = [];
	for (let i = 0; i < spriteCount; i++) {
		const x = i % columns;
		const y = (i - x) / columns;
		
		const inputFrame = cropImage(spritesheet, x, y, frameWidth, frameHeight);
		const targetFrame = cropImage(spritesheet, x + frameWidth, y, frameWidth, frameHeight);

		const input = await imageToTensor(inputFrame);
		const target = await imageToTensor(targetFrame);

		dataset.push({ input, target });
	}

	return dataset;
};

export const getInputs = (dataset) => dataset.map(instance => instance.input);
export const getTargets = (dataset) => dataset.map(instance => instance.target);
export const generateNoise = () => tf.randomNormal([batchSize, inputShape[0], inputShape[1], inputShape[2]]);
