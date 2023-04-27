<script>
	import * as tf from "@tensorflow/tfjs";
	import { spriteWidth, spriteHeight, spriteChannels } from "$lib/stores/data.js";
	import { generator, inputShape } from "$lib/stores/model.js";
	import { imageToSprite, spriteToInput, outputToSprite } from "$lib/utils/data.utils.js";
	
	let generating = false;
	let sourceImage;
	let generatedImage;
	let source;
	let white = false;

	const uploadSource = async (event) => {
		const sprite = await imageToSprite(URL.createObjectURL(event.target.files[0]), $spriteChannels);
		await tf.browser.toPixels(sprite, sourceImage);
		source = spriteToInput(sprite, $inputShape);
		sprite.dispose();
	};
	
	const generate = async (event) => {
		generating = true;
		const target = await $generator.predict(source);
		const targetSprite = outputToSprite(target, [$spriteHeight, $spriteWidth, $spriteChannels]);
		// const targetSprite = outputToSprite(source, [$spriteHeight, $spriteWidth, $spriteChannels]); // Testing sprite -> input/output -> sprite
		await tf.browser.toPixels(targetSprite, generatedImage);
		tf.dispose([target, targetSprite]);
		generating = false;
	};
	
	const toggleBackground = (event) => {
		white = !white;
	};
</script>

<section>
	<h2>Generate</h2>
	<input on:change={uploadSource} type="file" name="sourceImage">
	<p>Images must be PNG format and {$spriteWidth}px in width by {$spriteHeight}px in height.</p>
	{#if source}
		<button on:click={generate} disabled={!$generator || generating}>Generate</button>
		<button on:click={toggleBackground}>Toggle Background</button>
	{/if}
	<span>
		<canvas class:white bind:this={sourceImage}></canvas>
		<canvas class:white bind:this={generatedImage}></canvas>
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

	canvas.white {
		background-color: white;
	}

	button {
		width: fit-content;
		padding: 0.5em 1em;
	}
		
	input[type="file"] {
		cursor: pointer;
	}
</style>
