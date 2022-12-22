// Data Config
export const imageWidth = 64;
export const imageHeight = 64;
export const imageChannels = 4;	// RGBA

// RPG Maker 2000 Spritesheet
export const fileName = "RPG Maker 2000 Dedupe.png";
export const frameWidth = 24;
export const frameHeight = 32;
export const spriteCount = 71;	// Pairs of frames

// // Odyssey Spritesheet
// export const fileName = "Odyssey.png";
// export const frameWidth = 32;
// export const frameHeight = 32;
// export const spriteCount = 80;

export const datasetUrl = `/data/${fileName}`;
export const testInstances = 1;

// Model Config
export const inputShape = [imageWidth, imageHeight, imageChannels];
export const optimizer = "adam";
export const loss = "binaryCrossentropy";
export const metrics = ["accuracy"];
export const patchSize = [2, 2];
export const kernelSize = [4, 4];
export const padding = "same";

// Training Config
export const batchSize = 1;
export const epochs = 100;
