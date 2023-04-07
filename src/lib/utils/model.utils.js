import * as tf from "@tensorflow/tfjs";

export const createGenerator = (inputShape) => {
	const filters = 64;

	// Define the encoder part of the generator
	const input = tf.layers.input({ shape: inputShape, name: "generator_input" });
	
	let layer = input;
	for (let i = 0; i < 5; i++) {
		layer = tf.layers.conv2d({ filters: filters * (2 ** i), kernelSize: 4, strides: 2, padding: "same", activation: "relu" }).apply(layer);
		layer = tf.layers.layerNormalization().apply(layer);
		layer = tf.layers.leakyReLU({ alpha: 0.2 }).apply(layer);
	}

	// Define the decoder part of the generator with skip connections
	let skipLayers = [layer];
	for (let i = 4; i >= 0; i--) {
		const skipLayer = skipLayers[4 - i];
		layer = tf.layers.concatenate().apply([layer, skipLayer]);
		layer = tf.layers.conv2dTranspose({ filters: filters * (2 ** i), kernelSize: 4, strides: 2, padding: "same", activation: "relu" }).apply(layer);
		skipLayers.push(layer);
		layer = tf.layers.layerNormalization().apply(layer);
		layer = tf.layers.dropout(0.5).apply(layer);
		layer = tf.layers.reLU({ alpha: 0.2 }).apply(layer);
	}
	const output = tf.layers.conv2dTranspose({ filters: inputShape[2], kernelSize: 4, strides: 1, padding: "same", activation: "tanh", name: "generator_output" }).apply(layer);

	const generator = tf.model({ inputs: input, outputs: output, name: "generator" });		
	return generator;
};

export const createDiscriminator = (inputShape) => {
	const filters = [64, 128, 256, 512];

	// Define the model
	const inputSource = tf.layers.input({ shape: inputShape, name: "discriminator_input_source" });
	const inputTarget = tf.layers.input({ shape: inputShape, name: "discriminator_input_target" });
	const inputCombined = tf.layers.concatenate({ axis: 3 }).apply([inputSource, inputTarget]);
	
	let layer = inputCombined;
	// for (let i = 0; i < filters.length; i++) {
	for (let i = 0; i < 1; i++) {
		layer = tf.layers.conv2d({ filters: filters[i], kernelSize: 4, strides: 2, padding: "same", useBias: false }).apply(layer);
		// layer = tf.layers.batchNormalization().apply(layer);
		layer = tf.layers.leakyReLU({ alpha: 0.2 }).apply(layer);
	}

	// Output
	const output = tf.layers.conv2d({ filters: 1, kernelSize: 1, strides: 1, padding: "valid", useBias: false, activation: "sigmoid", name: "discriminator_output" }).apply(layer);

	const discriminator = tf.model({ inputs: [inputSource, inputTarget], outputs: output, name: "discriminator" });
	return discriminator;
};

export const createGAN = (inputShape, generator, discriminator) => {
	const inputSource = tf.input({ shape: inputShape });
	const genOut = generator.apply(inputSource);
	const discOut = discriminator.apply([inputSource, genOut]);
	const gan = tf.model({ inputs: inputSource, outputs: discOut });
	return gan;
};

export const generatorLoss = (discFakeOutput, fakeOutput, realOutput) => {
	const genLoss = tf.losses.sigmoidCrossEntropy(tf.onesLike(discFakeOutput), discFakeOutput);
	const l1Loss = tf.losses.absoluteDifference(realOutput, fakeOutput);
	const LAMBDA = 100;
	const loss = genLoss.add(l1Loss.mul(LAMBDA));
	return loss.mean();
};

export const discriminatorLoss = (realOutput, fakeOutput) => {
	const realLoss = tf.losses.sigmoidCrossEntropy(tf.onesLike(realOutput), realOutput);
	const fakeLoss = tf.losses.sigmoidCrossEntropy(tf.zerosLike(fakeOutput), fakeOutput);
	const loss = realLoss.add(fakeLoss);
	return loss.mean();
};

export const downloadModel = (model, name) => {
	model.save(`downloads://${name}`);
};
