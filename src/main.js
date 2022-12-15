import { loadDataset, getInputs, getTargets } from "./data.js";
import { createGAN, trainModel, testModel, generateImage } from "./model.js";

let model = createGAN();
const dataset = await loadDataset();
const testInstance = dataset.shift();
const testX = testInstance.input;
const testY = testInstance.target;
const trainX = getInputs(dataset);
const trainY = getTargets(dataset);

// UI
const resetButton = document.getElementById("reset");
const loadButton = document.getElementById("load");
const saveButton = document.getElementById("save");
const trainButton = document.getElementById("train");
const testButton = document.getElementById("test");
const generateButton = document.getElementById("generate");
const inputImage = document.getElementById("input");
const targetImage = document.getElementById("target");
const outputImage = document.getElementById("output");

let trained = false;
let training = false;

const onClickReset = (event) => {
	event.preventDefault();
	if (training) {
		console.log("Model is currently training...");
		return;
	}

	model = createGAN();
	trained = false;
};

const onClickLoad = async (event) => {
	event.preventDefault();
	if (training) {
		console.log("Model is currently training...");
		return;
	}

	const isSaved = localStorage.getItem("gan-model");
	if (isSaved) {
		console.log("Loading model...");
		model = await tf.loadLayersModel("localstorage://gan-model");
		console.log("Model loaded.");
	}
	else {
		console.log("No saved model found.");
	}
};

const onClickSave = async (event) => {
	event.preventDefault();
	if (training) {
		console.log("Model is currently training...");
		return;
	}

	if (!trained) {
		console.log("Model must be trained before saving.");
		return;
	}

	console.log("Saving model...");
	await model.save("localstorage://gan-model");
	console.log("Model saved.");
};

const onClickTrain = async (event) => {
	event.preventDefault();
	if (training) {
		console.log("Model is currently training...");
		return;
	}

	training = true;
	trained = false;
	const result = await trainModel(model, trainX, trainY);
	console.log(`Training Loss: ${result.history.loss[0]}`);
	trained = true;
	training = false;
};

const onClickTest = async (event) => {
	event.preventDefault();
	if (training) {
		console.log("Model is currently training...");
		return;
	}

	if (!trained) {
		console.log("Model must be trained before testing.");
		return;
	}

	const result = await testModel(model, testX, testY);
	result.print();
	// const prediction = result.dataSync();
	// console.log(`Testing Prediction: ${prediction}`);
	// Compare prediction to testY for accuracy
};

const onClickGenerate = async (event) => {
	event.preventDefault();
	if (training) {
		console.log("Model is currently training...");
		return;
	}

	if (!trained) {
		console.log("Model must be trained before generating.");
		return;
	}

	const prediction = await generateImage(model, testX, testY);
	console.log(prediction);
};

resetButton.addEventListener("click", onClickReset);
loadButton.addEventListener("click", onClickLoad);
saveButton.addEventListener("click", onClickSave);
trainButton.addEventListener("click", onClickTrain);
testButton.addEventListener("click", onClickTest);
generateButton.addEventListener("click", onClickGenerate);
