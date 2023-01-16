// Data Config
const imageWidth = 64;
const imageHeight = 64;
const imageChannels = 4;	// RGBA

const testInstances = 1;

// Model Config
const inputShape = [imageWidth, imageHeight, imageChannels];
const optimizer = "adam";
const loss = "binaryCrossentropy";
const metrics = ["accuracy"];
const patchSize = [2, 2];
const kernelSize = [4, 4];
const padding = "same";

// Training Config
const batchSize = 1;
const epochs = 100;
