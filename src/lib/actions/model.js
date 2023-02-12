import * as tf from "@tensorflow/tfjs";

export const createGenerator = (inputShape) => {
	const model = tf.sequential();

	// Encoder
	// 64x64x4
	model.add(tf.layers.conv2d({ filters: 64, kernelSize: 4, strides: 2, padding: "same", activation: "relu", inputShape }));
	model.add(tf.layers.leakyReLU());
	// 32x32x64
	model.add(tf.layers.conv2d({ filters: 128, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.leakyReLU());
	// 16x16x128
	model.add(tf.layers.conv2d({ filters: 256, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.leakyReLU());
	// 8x8x256
	model.add(tf.layers.conv2d({ filters: 512, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.leakyReLU());
	// 4x4x512
	model.add(tf.layers.conv2d({ filters: 512, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.leakyReLU());
	// 2x2x512
	model.add(tf.layers.conv2d({ filters: 512, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.leakyReLU());
	// 1x1x512

	// Decoder
	model.add(tf.layers.conv2dTranspose({ filters: 1024, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.dropout({ rate: 0.5 }));
	model.add(tf.layers.reLU());
	// 2x2x1024
	model.add(tf.layers.conv2dTranspose({ filters: 1024, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.dropout({ rate: 0.5 }));
	model.add(tf.layers.reLU());
	// 4x4x1024
	model.add(tf.layers.conv2dTranspose({ filters: 512, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.dropout({ rate: 0.5 }));
	model.add(tf.layers.reLU());
	// 8x8x512
	model.add(tf.layers.conv2dTranspose({ filters: 256, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.reLU());
	// 16x16x256
	model.add(tf.layers.conv2dTranspose({ filters: 128, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.reLU());
	// 32x32x128
	model.add(tf.layers.conv2dTranspose({ filters: 64, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.reLU());
	// 64x64x64
	model.add(tf.layers.conv2dTranspose({ filters: 32, kernelSize: 4, strides: 2, padding: "same", activation: "relu" }));
	model.add(tf.layers.layerNormalization());
	model.add(tf.layers.reLU());
	// 128x128x32

	// Output layer
	model.add(tf.layers.conv2d({ filters: 4, kernelSize: 4, strides: 2, padding: "same", activation: "tanh" }));
	// 64x64x4

	// Generator model is not compiled as it is only trained as part of the combined model
	// model.summary();
	return model;
};

export const createDiscriminator = (inputShape) => {
	const model = tf.sequential();

	model.add(tf.layers.conv2d({ filters: 64, kernelSize: 4, strides: 2, padding: 'same', inputShape }));
	model.add(tf.layers.leakyReLU({ alpha: 0.2 }));
	model.add(tf.layers.dropout({ rate: 0.3 }));

	model.add(tf.layers.conv2d({ filters: 128, kernelSize: 4, strides: 2, padding: 'same' }));
	model.add(tf.layers.batchNormalization());
	model.add(tf.layers.leakyReLU({ alpha: 0.2 }));
	model.add(tf.layers.dropout({ rate: 0.3 }));

	model.add(tf.layers.conv2d({ filters: 256, kernelSize: 4, strides: 2, padding: 'same' }));
	model.add(tf.layers.batchNormalization());
	model.add(tf.layers.leakyReLU({ alpha: 0.2 }));
	model.add(tf.layers.dropout({ rate: 0.3 }));

	model.add(tf.layers.flatten());
	model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

	// Discriminator model is compiled separately as it is only trained independently of the combined model
	const learningRate = 0.0002;
	const beta1 = 0.5;
	const optimizer = tf.train.adam(learningRate, beta1);
	const loss = "binaryCrossentropy";
	const metrics = ["accuracy"];
	model.compile({ loss, optimizer, metrics });
	// model.summary();
	return model;
};

export const createGan = (generator, discriminator) => {
	// Discriminator weights are frozen while the combined model trains
	// This allows the generator to apply the gradient of the discriminator
	discriminator.trainable = false;

	const model = tf.sequential();
	model.add(generator);
	model.add(discriminator);

	// The GAN is compiled with a loss function and optimizer
	const learningRate = 0.0002;
	const beta1 = 0.5;
	const optimizer = tf.train.adam(learningRate, beta1);
	const loss = "binaryCrossentropy";
	const metrics = ["accuracy"];
	model.compile({ loss, optimizer, metrics });
	return model;
};

export const downloadModel = (model, name) => {
	model.save(`downloads://${name}`);
};
