import { inputShape, optimizer, loss, patchSize, kernelSize, padding, imageChannels } from "./config.js";

/* U-Net Generator */

// export const createGenerator = () => {
// 	// Build the generator model
// 	const model = tf.sequential();

// 	// Encoder layers
// 	model.add(tf.layers.conv2d({ filters: 64, activation: "relu", kernelSize, strides: patchSize, padding, inputShape }));
// 	model.add(tf.layers.conv2d({ filters: 128, activation: "relu", kernelSize, strides: patchSize, padding }));
// 	model.add(tf.layers.conv2d({ filters: 256, activation: "relu", kernelSize, strides: patchSize, padding }));
// 	model.add(tf.layers.conv2d({ filters: 512, activation: "relu", kernelSize, strides: patchSize, padding }));

// 	// Decoder layers
// 	model.add(tf.layers.conv2dTranspose({ filters: 256, activation: "relu", kernelSize, strides: patchSize, padding }));
// 	model.add(tf.layers.conv2dTranspose({ filters: 128, activation: "relu", kernelSize, strides: patchSize, padding }));

// 	// Final output layer
// 	model.add(tf.layers.conv2dTranspose({ filters: imageChannels, activation: "tanh", kernelSize, strides: patchSize, padding }));
	
// 	// Compile the model
// 	model.compile({ optimizer, loss });

// 	return model;
// };

export const createGenerator = () => {
	const generator = tf.sequential();

	// Encoder
	generator.add(tf.layers.conv2d({ inputShape, kernelSize: 4, filters: 32, strides: 2, padding: 'same', activation: 'relu' }));
	generator.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
	generator.add(tf.layers.conv2d({ kernelSize: 4, filters: 64, strides: 1, padding: 'same', activation: 'relu' }));
	generator.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
	generator.add(tf.layers.conv2d({ kernelSize: 4, filters: 128, strides: 1, padding: 'same', activation: 'relu' }));
	generator.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
	
	// Decoder
	generator.add(tf.layers.conv2dTranspose({ kernelSize: 4, filters: 128, strides: 2, padding: 'same', activation: 'relu' }));
	generator.add(tf.layers.conv2dTranspose({ kernelSize: 4, filters: 64, strides: 2, padding: 'same', activation: 'relu' }));
	generator.add(tf.layers.conv2dTranspose({ kernelSize: 4, filters: 32, strides: 2, padding: 'same', activation: 'relu' }));
	
	// Output layer
	generator.add(tf.layers.conv2d({ kernelSize: 4, filters: 4, strides: 1, padding: 'same', activation: 'sigmoid' }));
	
	// Compile the model
	const optimizer = tf.train.adam(0.0002, 0.5);
	generator.compile({ optimizer, loss: 'binaryCrossentropy' });
	return generator;
};
