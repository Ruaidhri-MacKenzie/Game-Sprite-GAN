<script>
	import * as tf from "@tensorflow/tfjs";
	import { dataset, imageWidth, imageHeight, imageChannels, spriteWidth, spriteHeight, spriteChannels } from "$lib/stores/data.js";

	let spritesheet;
	$: spritesheetShape = spritesheet ? spritesheet.shape : null;
	$: spritesheetMin = spritesheet ? tf.min(spritesheet).dataSync() : null;
	$: spritesheetMax = spritesheet ? tf.max(spritesheet).dataSync() : null;

	let sprites;
	$: spritesShape = sprites ? sprites.shape : null;
	$: spritesMin = sprites ? tf.min(sprites).dataSync() : null;
	$: spritesMax = sprites ? tf.max(sprites).dataSync() : null;

	let loadingData = false;

	const onUploadSpritesheet = async (event) => {
		const image = new Image();
		image.src = URL.createObjectURL(event.target.files[0]);
    await new Promise((resolve, reject) => {
			image.onload = () => resolve(image);
			image.onerror = reject;
    });
		spritesheet = tf.browser.fromPixels(image, $spriteChannels);
	};

	const loadSpritesheet = async (event) => {
		// Turn spritesheet into dataset { source: tensor4d, target: tensor4d }
		loadingData = true;

		// Normalise values = [-1, 1] to match tanh output of generator
		sprites = spritesheet.div(255 / 2).sub(1);

		// Split into tensor of individual images
		const spriteCount = sprites.shape[1] / $spriteWidth;
		sprites = sprites.reshape([$spriteHeight, spriteCount, $spriteWidth, $spriteChannels]);
		sprites = sprites.transpose([1, 0, 2, 3]);
		sprites = sprites.reshape([spriteCount, $spriteHeight, $spriteWidth, $spriteChannels]);

		// Pad
		sprites = sprites.pad([[0, 0], [0, $imageHeight - $spriteHeight], [0, $imageWidth - $spriteWidth], [0, $imageChannels - $spriteChannels]], 0);

		// Crop
		sprites.slice([0, 0, 0, 0], [-1, $imageHeight, $imageWidth, $imageChannels]);

		// Split into arrays of source and target images
		const evenIndices = Array(sprites.shape[0]).fill(0).map((value, index) => index % 2 === 0 ? index : null).filter(value => value != null);
		const oddIndices = Array(sprites.shape[0]).fill(0).map((value, index) => index % 2 === 1 ? index : null).filter(value => value != null);
		$dataset = {
			source: sprites.gather(evenIndices),
			target: sprites.gather(oddIndices),
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
