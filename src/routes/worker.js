importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs");

const createGenerator = (inputShape) => {
	const filters = 64;

	// Define the encoder part of the generator
	const input = tf.layers.input({ shape: inputShape, name: "generator_input" });
	
	let layer = input;
	for (let i = 0; i < 5; i++) {
		layer = tf.layers.conv2d({ filters: filters * (2 ** i), kernelSize: 4, strides: 2, padding: "same", activation: "relu" }).apply(layer);
	}

	// Define the decoder part of the generator with skip connections
	let skipLayers = [layer];
	for (let i = 4; i >= 0; i--) {
		const skipLayer = skipLayers[4 - i];
		layer = tf.layers.concatenate().apply([layer, skipLayer]);
		layer = tf.layers.conv2dTranspose({ filters: filters * (2 ** i), kernelSize: 4, strides: 2, padding: "same", activation: "relu" }).apply(layer);
		skipLayers.push(layer);
	}
	const output = tf.layers.conv2dTranspose({ filters: inputShape[2], kernelSize: 4, strides: 1, padding: "same", activation: "tanh", name: "generator_output" }).apply(layer);

	const generator = tf.model({ inputs: input, outputs: output, name: "generator" });

	// Compile the model
	const learningRate = 0.0002;
	const optimizer = tf.train.adam({ learningRate });
	const loss = "binaryCrossentropy";
	const metrics = ["accuracy"];
	generator.compile({ optimizer, loss, metrics });

	return generator;
};

const createDiscriminator = (inputShape) => {
	const filters = 64;

	// Define the model
	const inputSource = tf.layers.input({ shape: inputShape, name: "discriminator_input_source" });
	const inputTarget = tf.layers.input({ shape: inputShape, name: "discriminator_input_target" });
	const inputCombined = tf.layers.concatenate({ axis: 3 }).apply([inputSource, inputTarget]);
	
	let layer = inputCombined;
	for (let i = 0; i < 1; i++) {
		layer = tf.layers.conv2d({ filters: filters * (2 ** i), kernelSize: 4, strides: 2, padding: "same", activation: "relu", alpha: 0.2 }).apply(layer); // leakyRelu?
	}

	const output = tf.layers.conv2d({ filters: 1, kernelSize: 2, strides: 1, padding: "valid", activation: "sigmoid", name: "discriminator_output" }).apply(layer);
	const discriminator = tf.model({ inputs: [inputSource, inputTarget], outputs: output, name: "discriminator" });

	// Compile the model
	const learningRate = 0.0002;
	const optimizer = tf.train.adam({ learningRate });
	const loss = "binaryCrossentropy";
	const metrics = ["accuracy"];
	discriminator.compile({ optimizer, loss, metrics });

	return discriminator;
};

const generator = createGenerator([64, 64, 4]);
const discriminator = createDiscriminator([64, 64, 4]);
const genOptimizer = tf.train.adam(0.0002, 0.5);
const discOptimizer = tf.train.adam(0.0002, 0.5);

const generatorLoss = (discGeneratedOutput, genOutput, target) => {
	const ganLoss = tf.losses.sigmoidCrossEntropy(tf.onesLike(discGeneratedOutput), discGeneratedOutput);
	const l1Loss = tf.losses.absoluteDifference(target, genOutput);
	const totalLoss = ganLoss.add(l1Loss.mul(100));
	// return totalLoss;
	return ganLoss;
};

const discriminatorLoss = (discRealOutput, discGeneratedOutput) => {
	const realLoss = tf.losses.sigmoidCrossEntropy(discRealOutput, tf.onesLike(discRealOutput));
	const generatedLoss = tf.losses.sigmoidCrossEntropy(discGeneratedOutput, tf.zerosLike(discGeneratedOutput));
	const totalLoss = realLoss.add(generatedLoss);
	return totalLoss;
};

const trainModel = (sources, targets, steps) => {
	try {
		let startTime = Date.now();
		sources = tf.tensor4d(sources, [80, 64, 64, 4]);
		targets = tf.tensor4d(targets, [80, 64, 64, 4]);

		for (let step = 0; step < steps; step++) {
			const source = sources.slice([step % sources.shape[0], 0, 0, 0], [1, -1, -1, -1]);
			const target = targets.slice([step % targets.shape[0], 0, 0, 0], [1, -1, -1, -1]);
			if (step % 1000 === 0) {
				if (step !== 0) {
					console.log(`Time taken for 1000 steps: ${((Date.now() - startTime) / 1000).toFixed(2)} sec`);
				}
				startTime = Date.now();
				console.log(`Step: ${step / 1000}k`);
			}

			tf.tidy(() => {
				// Train Generator
				const genTape = tf.variableGrads(() => {
					const genOutput = generator.predict(source);
					const discGeneratedOutput = discriminator.predict([source, genOutput]);
					const genLoss = generatorLoss(discGeneratedOutput, genOutput, target);
					return genLoss;
				});
				genOptimizer.applyGradients(genTape.grads);
				
				// Train Discriminator
				const discTape = tf.variableGrads(() => {
					const discRealOutput = discriminator.predict([source, target]);
					const genOutput = generator.predict(source);
					const discGeneratedOutput = discriminator.predict([source, genOutput]);
					const discLoss = discriminatorLoss(discRealOutput, discGeneratedOutput);
					return discLoss;
				});
				discOptimizer.applyGradients(discTape.grads);

				if (step % 10 === 0) {
					postMessage({ step, genLoss: genTape.value.dataSync()[0], discLoss: discTape.value.dataSync()[0] });
				}
			});

			// Save model weights (checkpoint) every 5k steps
			if ((step + 1) % 5000 === 0) {
				// checkpoint.save(file_prefix=CHECKPOINT_PREFIX);
			}
			source.dispose();
			target.dispose();
		}
		
		postMessage({ success: true });
	}
	catch (error) {
		postMessage({ error });
	}
};

const generate = async (source) => {
	source = tf.tensor4d(source, [1, 64, 64, 4]);
	const target = await generator.predict(source);
	postMessage({ target: target.dataSync() });
};

self.addEventListener("message", (event) => {
	if (event.data.source) {
		generate(event.data.source);
	}
	else {
		const { sources, targets, steps } = event.data;
		trainModel(sources, targets, steps);
	}
});
