import { inputShape, optimizer, loss, patchSize, kernelSize, padding, imageChannels } from "./config.js";

/* U-Net Generator */

export const createGenerator = () => {
	// Build the generator model
	const model = tf.sequential();

	// Encoder layers
	model.add(tf.layers.conv2d({ filters: 64, activation: "relu", kernelSize, strides: patchSize, padding, inputShape }));
	model.add(tf.layers.conv2d({ filters: 128, activation: "relu", kernelSize, strides: patchSize, padding }));
	model.add(tf.layers.conv2d({ filters: 256, activation: "relu", kernelSize, strides: patchSize, padding }));
	model.add(tf.layers.conv2d({ filters: 512, activation: "relu", kernelSize, strides: patchSize, padding }));

	// Decoder layers
	model.add(tf.layers.conv2dTranspose({ filters: 256, activation: "relu", kernelSize, strides: patchSize, padding }));
	model.add(tf.layers.conv2dTranspose({ filters: 128, activation: "relu", kernelSize, strides: patchSize, padding }));

	// Final output layer
	model.add(tf.layers.conv2dTranspose({ filters: imageChannels, activation: "sigmoid", kernelSize, strides: patchSize, padding }));
	
	// Compile the model
	model.compile({ optimizer, loss });

	return model;
};
