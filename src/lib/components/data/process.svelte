<script>
	import * as tf from "@tensorflow/tfjs";
	import { trainData, testData, testInputs, spriteWidth, spriteHeight, spriteChannels } from "$lib/stores/data.js";
	import { inputShape } from "$lib/stores/model.js";
	import { testSources, testTargets } from "$lib/stores/train.js";
	import { getBatch, outputsToImages, normalisePixelValues, splitSpritesheet, padInput, cropInput } from "$lib/utils/data.utils.js";
	import TensorInfo from "./tensor-info.svelte";

	export let spritesheet;
	export let sprites;

	let loadingData = false;
	let trainTestSplit = 3;

	const processSpritesheet = async (event) => {
		// Turn spritesheet into test and train datasets { source: tensor4d, target: tensor4d }
		loadingData = true;

		// Normalise values = [-1, 1] to match tanh output of generator
		sprites = normalisePixelValues(spritesheet);

		// Split into tensor of individual images (3d tensor to 4d tensor)
		sprites = splitSpritesheet(sprites, [$spriteHeight, $spriteWidth, $spriteChannels]);

		// Pad image to input shape
		sprites = padInput(sprites, [$inputShape[0] - $spriteHeight, $inputShape[1] - $spriteWidth, $inputShape[2] - $spriteChannels]);

		// Crop image to input shape
		sprites = cropInput(sprites, $inputShape);

		// Split into arrays of source and target images
		const evenIndices = Array(sprites.shape[0]).fill(0).map((value, index) => index % 2 === 0 ? index : null).filter(value => value != null);
		const oddIndices = Array(sprites.shape[0]).fill(0).map((value, index) => index % 2 === 1 ? index : null).filter(value => value != null);
		const source = sprites.gather(evenIndices);
		const target = sprites.gather(oddIndices);
		// const source = sprites.gather(oddIndices);
		// const target = sprites.gather(evenIndices);

		// const [jitterSource, jitterTarget] = applyBatchJitter(source, target);
		
		// Split into test and train datasets
		const pairCount = source.shape[0];
		const testCount = trainTestSplit >= 1 ? Math.floor(trainTestSplit) : Math.floor(pairCount * (1 - trainTestSplit));

		$trainData = {
			source: getBatch(source, $inputShape, 0, pairCount - testCount),
			target: getBatch(target, $inputShape, 0, pairCount - testCount),
		};
		
		$testData = {
			source: getBatch(source, $inputShape, pairCount - testCount, testCount),
			target: getBatch(target, $inputShape, pairCount - testCount, testCount),
		};

		// Create list of source inputs for testing
		$testInputs = [
			getBatch($trainData.source, $inputShape, 0, 1),
			getBatch($testData.source, $inputShape, 0, 1),
			getBatch($testData.source, $inputShape, 1, 1),
			getBatch($testData.source, $inputShape, 2, 1),
		];

		// Create list of target outputs for testing
		const testOutputs = [
			getBatch($trainData.target, $inputShape, 0, 1),
			getBatch($testData.target, $inputShape, 0, 1),
			getBatch($testData.target, $inputShape, 1, 1),
			getBatch($testData.target, $inputShape, 2, 1),
		];

		const spriteShape = [$spriteHeight, $spriteWidth, $spriteChannels];

		// Convert source inputs to images
		$testSources = await outputsToImages($testInputs, spriteShape);

		// Convert target outputs to images
		$testTargets = await outputsToImages(testOutputs, spriteShape);

		tf.dispose([source, target, testOutputs]);
		loadingData = false;
	};
</script>

<form>
	<h3>Process</h3>

	<div>
		<label for="width">Width:</label>
		<input bind:value={$spriteWidth} type="number" id="width" min=1 step=1>

		<label for="height">Height:</label>
		<input bind:value={$spriteHeight} type="number" id="height" min=1 step=1>

		<label for="channels">Channels:</label>
		<select bind:value={$spriteChannels} id="channels">
			<option value={1}>Greyscale</option>
			<option value={3}>RGB</option>
			<option value={4}>RGBA</option>
		</select>
	</div>

	{#if loadingData}
		<button disabled>Loading...</button>
	{:else}
		<button disabled={!spritesheet} on:click={processSpritesheet}>Process Spritesheet</button>
	{/if}
	
	<TensorInfo tensor={sprites} />
</form>

<style>
	form {
		display: grid;
		gap: 1em;
	}

	h3 {
		text-align: center;
	}

	div {
		width: min-content;
		display: grid;
		grid-template-columns: auto 1fr;
		align-items: center;
		gap: 0.5em;
	}

	label {
		text-align: right;
	}

	input {
		width: 7.125em;
	}
</style>
