<script>
	import * as tf from "@tensorflow/tfjs";
	import { onMount } from "svelte";
	import { spriteHeight, spriteWidth, spriteChannels } from "$lib/stores/data.js";
	import { inputShape } from "$lib/stores/model.js";
	import { getSample, outputToSprite } from "$lib/utils/data.utils.js";

	export let dataset;
	let backgroundColour = "#000000";

	let sourcePreview;
	let targetPreview;
	let previewIndex = 0;
	$: spritePairCount = dataset ? dataset.source.shape[0] : 0;

	const renderPreview = () => {
		tf.tidy(() => {
			const source = outputToSprite(getSample(dataset.source, $inputShape, previewIndex), [$spriteHeight, $spriteWidth, $spriteChannels]);
			tf.browser.toPixels(source, sourcePreview);
			
			const target = outputToSprite(getSample(dataset.target, $inputShape, previewIndex), [$spriteHeight, $spriteWidth, $spriteChannels]);
			tf.browser.toPixels(target, targetPreview);
		});
	};

	const backPreview = () => {
		previewIndex = (previewIndex > 0) ? previewIndex - 1 : spritePairCount - 1;
		renderPreview();
	};
	
	const nextPreview = () => {
		previewIndex = (previewIndex < spritePairCount - 1) ? previewIndex + 1 : 0;
		renderPreview();
	};

	onMount(renderPreview);
</script>

<div>
	<h3>Preview</h3>

	<label>
		Background:
		<input type="color" bind:value={backgroundColour}>
	</label>

	<span style="--background: {backgroundColour}">
		<button on:click={backPreview}>Back</button>
		<canvas bind:this={sourcePreview}></canvas>
		<canvas bind:this={targetPreview}></canvas>
		<button on:click={nextPreview}>Next</button>
	</span>
	
	<p>Sprite {previewIndex + 1}/{spritePairCount}</p>
</div>

<style>
	div {
		width: fit-content;
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 1em;
	}
	
	h3 {
		text-align: center;
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

	span {
		display: flex;
	}

	button {
		padding: 0.25em 0.5em;
		border-radius: 0;
		cursor: pointer;
	}

	canvas {
		background-color: var(--background);
		width: 64px;
		height: 64px;
	}

	p {
		margin-inline: auto;
	}
</style>
