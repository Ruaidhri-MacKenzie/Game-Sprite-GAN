const spritesheets = [
	{
		fileName: "RPG Maker 2000 Dedupe.png",
		frameWidth: 24,
		frameHeight: 32,
		spriteCount: 71,	// Pairs of frames
	},
	{
		fileName: "Odyssey.png",
		frameWidth: 32,
		frameHeight: 32,
		spriteCount: 80,	// Pairs of frames
	},
];

const spritesheet = spritesheets[0];
const { fileName, frameWidth, frameHeight, spriteCount } = spritesheet;
const spritesheetUrl = `/data/${fileName}`;

const imageToTensor = async (image, channels) => {
	// const tensor = await tf.browser.fromPixelsAsync(image, channels);
	const tensor = tf.browser.fromPixels(image, channels);
	return tensor;
};

const tensorToImage = async (tensor) => {
	const canvas = new OffscreenCanvas(tensor.shape[0], tensor.shape[1]);
	await tf.browser.toPixels(tensor, canvas);
	const image = new Image();
	image.src = canvas.transferToImageBitmap();
	await new Promise((resolve, reject) => {
		image.onload = resolve;
		image.onerror = reject;
	});
	return image;
};

const loadImage = async (url) => {
	const image = new Image();
	image.src = url;
	await new Promise((resolve, reject) => {
		image.onload = resolve;
		image.onerror = reject;
	});
	return image;
};

// UI
const saveButton = document.getElementById("save");
const loadButton = document.getElementById("load");
const trainButton = document.getElementById("train");
const testButton = document.getElementById("test");
const inputImage = document.getElementById("input");
const targetImage = document.getElementById("target");
const outputImage = document.getElementById("output");

const onClickLoad = async (event) => {
	event.preventDefault();
	// GET /model
};

const onClickSave = async (event) => {
	event.preventDefault();
	// POST /model
};

const onClickTrain = async (event) => {
	event.preventDefault();
	// GET /model/train
};

const onClickTest = async (event) => {
	event.preventDefault();
	// GET /model/test
};

saveButton.addEventListener("click", onClickSave);
loadButton.addEventListener("click", onClickLoad);
trainButton.addEventListener("click", onClickTrain);
testButton.addEventListener("click", onClickTest);
