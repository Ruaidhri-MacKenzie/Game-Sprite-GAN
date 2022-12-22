import { inputShape, optimizer, loss, patchSize, kernelSize, padding } from "./config.js";

/* Patch GAN Discriminator */

// export const createDiscriminator = () => {
// 	// Build the discriminator model
// 	const model = tf.sequential();

// 	// Convolutional layer with a patch-size kernel
// 	model.add(tf.layers.conv2d({ filters: 64, activation: "relu", kernelSize, strides: patchSize, padding, inputShape }));
// 	model.add(tf.layers.conv2d({ filters: 64, activation: "relu", kernelSize, strides: patchSize, padding }));

// 	// Global average pooling layer
// 	model.add(tf.layers.globalAveragePooling2d({ poolSize: patchSize }));

// 	// Final output layer with a single unit
// 	model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

// 	// Compile the model
// 	model.compile({ optimizer, loss });

// 	return model;
// };

export const createDiscriminator = () => {
	const discriminator = tf.sequential();

	discriminator.add(tf.layers.conv2d({ inputShape, kernelSize: 3, filters: 32, strides: 2, padding: 'same', activation: 'relu' }));
	discriminator.add(tf.layers.dropout({ rate: 0.3 }));
	discriminator.add(tf.layers.conv2d({ kernelSize: 3, filters: 64, strides: 2, padding: 'same', activation: 'relu' }));
	discriminator.add(tf.layers.dropout({ rate: 0.3 }));
	discriminator.add(tf.layers.conv2d({ kernelSize: 3, filters: 128, strides: 2, padding: 'same', activation: 'relu' }));
	discriminator.add(tf.layers.dropout({ rate: 0.3 }));
	discriminator.add(tf.layers.flatten());
	
	// Output layer
	discriminator.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
	
	// Compile the model
	const optimizer = tf.train.adam(0.0002, 0.5);
	discriminator.compile({ optimizer: optimizer, loss: 'binaryCrossentropy' });
	return discriminator;
};
