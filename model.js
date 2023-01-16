const tf = require("@tensorflow/tfjs-node-gpu");

const createGenerator = () => {
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
	
	return generator;
};

const createDiscriminator = () => {
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
	
	return discriminator;
};

const createGan = (generator, discriminator) => {
	// Freeze the weights of the discriminator
	discriminator.trainable = false;
	
	// Build the combined model
	const combined = tf.sequential();
	combined.add(generator);
	combined.add(discriminator);
	return combined;
};

const generator = createGenerator();
const discriminator = createDiscriminator();
const gan = createGan(generator, discriminator);

const generatorOptimizer = tf.train.adam(0.0002, 0.5);
const discriminatorOptimizer = tf.train.adam(0.0002, 0.5);
const ganOptimizer = tf.train.adam(0.0002, 0.5);

module.exports = {
	generator,
	discriminator,
	gan,
	generatorOptimizer,
	discriminatorOptimizer,
	ganOptimizer,
};
