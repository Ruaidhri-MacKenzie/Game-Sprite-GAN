// Data Config
export const imageWidth = 64;
export const imageHeight = 64;
export const imageChannels = 4;

export const fileName = "RPG Maker 2000 Dedupe.png";
export const frameWidth = 24;
export const frameHeight = 32;
export const spriteCount = 71;

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
