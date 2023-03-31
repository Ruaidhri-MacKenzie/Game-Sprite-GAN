<script>
	import * as tf from "@tensorflow/tfjs";
	import { imageWidth, imageHeight, imageChannels, spriteWidth, spriteHeight, spriteChannels } from "$lib/stores/data.js";
	import { generator } from "$lib/stores/model.js";
	
	let generating = false;
	let sourceImage;
	let generatedImage;
	let source;

	const spriteToInput = (sprite) => {
		return tf.tidy(() => {
			// Normalise values to [-1,1]
			sprite = sprite.div(255 / 2).sub(1);
	
			// Pad to model input shape
			// sprite = sprite.pad([[0, $imageHeight - $spriteHeight], [0, $imageWidth - $spriteWidth], [0, $imageChannels - $spriteChannels]], 0);
	
			// Add batch dimension
			sprite = tf.expandDims(sprite);
	
			return sprite;
		});
	};

	const inputToSprite = (input) => {
		return tf.tidy(() => {
			// Crop to sprite shape
			input = input.slice([0, 0, 0, 0], [1, $spriteHeight, $spriteWidth, $spriteChannels]);
			
			// Remove batch dimension
			input = input.squeeze();
			
			// Normalise values to [0,1]
			input = input.add(1).div(2);
	
			return input;
		});
	};

	const uploadSource = async (event) => {
		const image = new Image();
		image.src = URL.createObjectURL(event.target.files[0]);
    await new Promise((resolve, reject) => {
			image.onload = () => resolve(image);
			image.onerror = reject;
    });
		const sprite = tf.browser.fromPixels(image, $spriteChannels);
		await tf.browser.toPixels(sprite, sourceImage);
		source = spriteToInput(sprite);
		sprite.dispose();
	};
	
	const generate = async (event) => {
		generating = true;
		const target = await $generator.predict(source);
		const targetSprite = inputToSprite(target);
		// const targetSprite = inputToSprite(source); // Testing sprite -> input/output -> sprite
		await tf.browser.toPixels(targetSprite, generatedImage);
		generating = false;
		tf.dispose([target, targetSprite]);
	};
</script>

<section>
	<h2>Generate</h2>
	<input on:change={uploadSource} type="file" name="sourceImage">
	{#if source}
		<button on:click={generate} disabled={!$generator || generating}>Generate</button>
	{/if}
	<span>
		<canvas bind:this={sourceImage}></canvas>
		<canvas bind:this={generatedImage}></canvas>
	</span>
</section>

<style>
	section {
		display: grid;
		gap: 1em;
		padding: 1em;
		box-shadow: 0 3px 8px hsl(0 0% 0% / 0.24);
	}

	span {
		display: flex;
		gap: 1em;
		padding: 1em;
	}

	canvas {
		background-color: black;
		width: 64px;
		height: 64px;
	}

	button {
		width: fit-content;
		padding: 0.5em 1em;
	}
		
	input[type="file"] {
		cursor: pointer;
	}
</style>
