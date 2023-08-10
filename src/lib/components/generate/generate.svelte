<script>
	import * as tf from "@tensorflow/tfjs";
	import { spriteWidth, spriteHeight, spriteChannels } from "$lib/stores/data.js";
	import { generator, inputShape } from "$lib/stores/model.js";
	import { imageToSprite, spriteToInput, outputToSprite } from "$lib/utils/data.utils.js";
	import Section from "$lib/components/common/section.svelte";
	import InputUpload from "$lib/components/common/input-upload.svelte";

	let generating = false;
	let sourceImage;
	let generatedImage;
	let source;
	let backgroundColour = "#000000";

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
</script>

<Section title="Generate">
	<div>
		<InputUpload text="Upload Source Image" handleChange={uploadSource} accept="image/*"/>
		<p>Images must be PNG format and {$spriteWidth}px in width by {$spriteHeight}px in height.</p>
		{#if source}
			<button on:click={generate} disabled={!$generator || generating}>Generate</button>
			<label>
				Background:
				<input type="color" bind:value={backgroundColour}>
			</label>
		{/if}
		<span style="--background: {backgroundColour}">
			<canvas bind:this={sourceImage}></canvas>
			<canvas bind:this={generatedImage}></canvas>
		</span>
	</div>
</Section>

<style>
	div {
		display: grid;
		gap: 1em;
		padding: 1em;
	}

	span {
		display: flex;
		gap: 1em;
		padding: 1em;
	}
	
	label {
		width: fit-content;
		display: flex;
		align-items: center;
		gap: 1em;
	}

	input {
		cursor: pointer;
	}

	canvas {
		background-color: var(--background);
		width: 64px;
		height: 64px;
	}
</style>
