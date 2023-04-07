<script>
	import * as tf from "@tensorflow/tfjs";
	import { trainData, testData, trainTestSplit, spriteWidth, spriteHeight, spriteChannels } from "$lib/stores/data.js";
	import { inputShape } from "$lib/stores/model.js";
	import { getBatch, imageToSprite, splitSpritesheet } from "$lib/utils/data.utils.js";

	let spritesheet;
	$: spritesheetShape = (spritesheet) ? spritesheet.shape : null;
	$: spritesheetMin = (spritesheet) ? tf.min(spritesheet).dataSync() : null;
	$: spritesheetMax = (spritesheet) ? tf.max(spritesheet).dataSync() : null;

	let sprites;
	$: spritesShape = (sprites) ? sprites.shape : null;
	$: spritesMin = (sprites) ? tf.min(sprites).dataSync() : null;
	$: spritesMax = (sprites) ? tf.max(sprites).dataSync() : null;

	let loadingData = false;

	const onUploadSpritesheet = async (event) => {
		spritesheet = await imageToSprite(URL.createObjectURL(event.target.files[0]), $spriteChannels);
	};

	const applyBatchJitter = ({ source, target }) => {
		return tf.tidy(() => {
			const batchSize = source.shape[0];
			const jitter = 0.1;
	
			// Apply random rotation within the jitter range
			const angles = tf.randomUniform([batchSize], -jitter, jitter);
			const rotatedSource = tf.image.rotateWithOffset(source, angles);
			const rotatedTarget = tf.image.rotateWithOffset(target, angles);
	
			// // Apply random translation within the jitter range
			// const tx = tf.randomUniform([batchSize], -jitter, jitter);
			// const ty = tf.randomUniform([batchSize], -jitter, jitter);
			// const transforms = tf.stack([
			// 	tf.ones([batchSize]), tf.zeros([batchSize]), tx,
			// 	tf.zeros([batchSize]), tf.ones([batchSize]), ty,
			// 	tf.zeros([batchSize]), tf.zeros([batchSize]), tf.ones([batchSize]),
			// ], 1);
			// const translatedSource = tf.image.transform(rotatedSource, transforms, "nearest");
			// const translatedTarget = tf.image.transform(rotatedTarget, transforms, "nearest");
	
			// // Apply random scaling within the jitter range
			// const scales = tf.randomUniform([batchSize], 1 - jitter, 1 + jitter);
			// const scaledSourceImages = tf.unstack(translatedSource).map((sprite, i) => tf.image.resizeBilinear(sprite, [scales[i] * 32, scales[i] * 32]));
			// const scaledSource = tf.stack(scaledSourceImages);
			// const scaledTargetImages = tf.unstack(translatedTarget).map((sprite, i) => tf.image.resizeBilinear(sprite, [scales[i] * 32, scales[i] * 32]));
			// const scaledTarget = tf.stack(scaledTargetImages);
	
			const newSource = source.concat(rotatedSource);
			const newTarget = target.concat(rotatedTarget);
			return { source: newSource, target: newTarget };
		});
	};

	const loadSpritesheet = async (event) => {
		// Turn spritesheet into test and train datasets { source: tensor4d, target: tensor4d }
		loadingData = true;

		// Normalise values = [-1, 1] to match tanh output of generator
		sprites = spritesheet.div(255 / 2).sub(1);

		// Split into tensor of individual images (3d tensor to 4d tensor)
		sprites = splitSpritesheet(sprites, [$spriteHeight, $spriteWidth, $spriteChannels]);

		// Pad image to input shape
		sprites = sprites.pad([[0, 0], [0, $inputShape[0] - $spriteHeight], [0, $inputShape[1] - $spriteWidth], [0, $inputShape[2] - $spriteChannels]], 0);

		// Crop image to input shape
		sprites.slice([0, 0, 0, 0], [-1, ...$inputShape]);

		// Split into arrays of source and target images
		const evenIndices = Array(sprites.shape[0]).fill(0).map((value, index) => index % 2 === 0 ? index : null).filter(value => value != null);
		const oddIndices = Array(sprites.shape[0]).fill(0).map((value, index) => index % 2 === 1 ? index : null).filter(value => value != null);
		const data = { source: sprites.gather(evenIndices), target: sprites.gather(oddIndices) };

		// const generatedData = applyBatchJitter(data);
		
		// Split into test and train datasets
		const pairCount = data.source.shape[0];
		const testCount = Math.floor(pairCount * (1 - $trainTestSplit));

		$trainData = {
			source: getBatch(data.source, $inputShape, 0, pairCount - testCount),
			target: getBatch(data.target, $inputShape, 0, pairCount - testCount),
		};
		
		$testData = {
			source: getBatch(data.source, $inputShape, pairCount - testCount, testCount),
			target: getBatch(data.target, $inputShape, pairCount - testCount, testCount),
		};

		loadingData = false;
	};
</script>

<form>
	<div>
		<label for="spritesheet">Spritesheet:</label>
		<input on:change={onUploadSpritesheet} type="file" id="spritesheet" accept="image/*">

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
	
	{#if spritesheet}
		<p>Shape: [{spritesheetShape}] Range: [{spritesheetMin},{spritesheetMax}]</p>
	{/if}

	{#if loadingData}
		<button disabled>Loading...</button>
	{:else}
		<button disabled={!spritesheet} on:click={loadSpritesheet}>Load Spritesheet</button>
	{/if}

	{#if sprites}
		<p>Shape: [{spritesShape}] Range: [{spritesMin},{spritesMax}]</p>
	{/if}
</form>

<style>
	form {
		display: grid;
		align-content: start;
		gap: 1em;
		padding-top: 0.5em;
	}
	
	div {
		width: fit-content;
		display: grid;
		grid-template-columns: auto 1fr;
		justify-items: start;
		align-items: center;
		gap: 0.5em;
	}

	label {
		text-align: right;
	}

	input[type="file"] {
		cursor: pointer;
	}

	select {
		padding: 0.25em;
		cursor: pointer;
	}
	
	button {
		width: fit-content;
		padding: 0.5em 1em;
		cursor: pointer;
	}

	button[disabled] {
		cursor: not-allowed;
	}
</style>
