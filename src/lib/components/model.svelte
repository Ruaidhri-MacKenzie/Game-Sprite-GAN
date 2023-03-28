<script>
	import * as tf from "@tensorflow/tfjs";
	import { imageWidth, imageHeight, imageChannels } from "$lib/stores/data.js";
	import { generator, discriminator } from "$lib/stores/model.js";

	const generatorOptimizer = tf.train.adam(0.0002);
	const discriminatorOptimizer = tf.train.adam(0.0002);
	
	let loading = false;
	$: status = (loading) ? "loading..." : ($generator && $discriminator) ? "✓" : "X";

	const createGenerator = () => {
		const inputShape = [$imageHeight, $imageWidth, $imageChannels];
		const filters = 64;

		// Define the encoder part of the generator
		const input = tf.layers.input({ shape: inputShape, name: "generator_input" });
		
		let layer = input;
		for (let i = 0; i < 4; i++) {
			layer = tf.layers.conv2d({ filters: filters * (2 ** i), kernelSize: 4, strides: 2, padding: "same", activation: "relu" }).apply(layer);
		}

		// Define the decoder part of the generator with skip connections
		let skipLayers = [layer];
		for (let i = 3; i >= 0; i--) {
			const skipLayer = skipLayers[3 - i];
			layer = tf.layers.concatenate().apply([layer, skipLayer]);
			layer = tf.layers.conv2dTranspose({ filters: filters * (2 ** i), kernelSize: 4, strides: 2, padding: "same", activation: "relu" }).apply(layer);
			skipLayers.push(layer);
		}
		const output = tf.layers.conv2dTranspose({ filters: inputShape[2], kernelSize: 4, strides: 1, padding: "same", activation: "tanh", name: "generator_output" }).apply(layer);

		const generator = tf.model({ inputs: input, outputs: output, name: "generator" });		
		return generator;
	};

	const createDiscriminator = () => {
		const inputShape = [$imageHeight, $imageWidth, $imageChannels];
		const filters = [64, 128, 256, 512];

		// Define the model
		const inputSource = tf.layers.input({ shape: inputShape, name: "discriminator_input_source" });
		const inputTarget = tf.layers.input({ shape: inputShape, name: "discriminator_input_target" });
		const inputCombined = tf.layers.concatenate({ axis: 3 }).apply([inputSource, inputTarget]);
		
		let layer = inputCombined;
		// for (let i = 0; i < filters.length; i++) {
		for (let i = 0; i < 1; i++) {
			layer = tf.layers.conv2d({ filters: filters[i], kernelSize: 4, strides: 2, padding: "same", useBias: false }).apply(layer);
			layer = tf.layers.batchNormalization().apply(layer);
			layer = tf.layers.leakyReLU({ alpha: 0.2 }).apply(layer);
		}

		// Output
		const output = tf.layers.conv2d({ filters: 1, kernelSize: 2, strides: 1, padding: "valid", useBias: false, activation: "sigmoid", name: "discriminator_output" }).apply(layer);

		const discriminator = tf.model({ inputs: [inputSource, inputTarget], outputs: output, name: "discriminator" });
		return discriminator;
	};

	const generatorLoss = (realOutput, fakeOutput) => {
		const genLoss = tf.losses.sigmoidCrossEntropy(tf.onesLike(fakeOutput), fakeOutput);
		const l1Loss = tf.losses.absoluteDifference(realOutput, fakeOutput);
		const LAMBDA = 100;
		const loss = genLoss.add(l1Loss.mul(LAMBDA));
		return loss.mean();
	};

	const discriminatorLoss = (realOutput, fakeOutput) => {
		const realLoss = tf.losses.sigmoidCrossEntropy(tf.onesLike(realOutput), realOutput);
		const fakeLoss = tf.losses.sigmoidCrossEntropy(tf.zerosLike(fakeOutput), fakeOutput);
		const loss = realLoss.add(fakeLoss);
		return loss.mean();
	};
	
	const createModels = (event) => {
		loading = true;
		
		$generator = createGenerator();
		$generator.compile({ optimizer: generatorOptimizer, loss: "meanSquaredError" });
		// $generator.compile({ optimizer: generatorOptimizer, loss: "binaryCrossentropy" });
		// $generator.compile({ optimizer: generatorOptimizer, loss: generatorLoss });
		// $generator.compile({ optimizer: generatorOptimizer, loss: "binaryCrossentropy", metrics: ["accuracy"] });
		$generator.summary();
		
		$discriminator = createDiscriminator();
		$discriminator.compile({ optimizer: discriminatorOptimizer, loss: "binaryCrossentropy" });
		// $discriminator.compile({ optimizer: discriminatorOptimizer, loss: discriminatorLoss });
		// $discriminator.compile({ optimizer: discriminatorOptimizer, loss: "binaryCrossentropy", metrics: ["accuracy"] });
		$discriminator.summary();
		
		loading = false;
	};

	const downloadModel = (model, name) => {
		model.save(`downloads://${name}`);
	};

	const saveModels = (event) => {
		if ($generator) downloadModel($generator, "generator");
		if ($discriminator) downloadModel($discriminator, "discriminator");
	};
</script>

<section>
	<h2>Model</h2>
	<button on:click={createModels}>Create Models</button>
	<button on:click={saveModels}>Save Models</button>
	<p>Generator: <span class:active={$generator}>{status}</span></p>
	<p>Discriminator: <span class:active={$discriminator}>{status}</span></p>
</section>

<style>
	section {
		display: grid;
		gap: 1em;
		padding: 1em;
		box-shadow: 0 3px 8px hsl(0 0% 0% / 0.24);
	}

	button {
		width: fit-content;
		padding: 0.25em 0.5em;
	}

	span {
		color: red;
	}

	span.active {
		color: green;
	}
</style>
