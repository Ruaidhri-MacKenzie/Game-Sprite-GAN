import { inputShape, optimizer, loss, patchSize, kernelSize, padding } from "./config.js";

/* Patch GAN Discriminator */

export const createDiscriminator = () => {
	// Build the discriminator model
	const model = tf.sequential();

	// Convolutional layer with a patch-size kernel
	model.add(tf.layers.conv2d({ filters: 64, activation: "relu", kernelSize, strides: patchSize, padding, inputShape }));
	model.add(tf.layers.conv2d({ filters: 64, activation: "relu", kernelSize, strides: patchSize, padding }));

	// Global average pooling layer
	model.add(tf.layers.globalAveragePooling2d({ poolSize: patchSize }));

	// Final output layer with a single unit
	model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

	// Compile the model
	model.compile({ optimizer, loss });

	return model;
};
