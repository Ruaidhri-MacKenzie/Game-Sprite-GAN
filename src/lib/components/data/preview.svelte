<script>
	import * as tf from "@tensorflow/tfjs";
	import { onMount } from "svelte";
	import { trainData, spriteHeight, spriteWidth, spriteChannels } from "$lib/stores/data.js";
	import { inputShape } from "$lib/stores/model.js";
	import { getSample, outputToSprite } from "$lib/utils/data.utils.js";

	let sourcePreview;
	let targetPreview;
	let previewIndex = 0;
	$: spritePairCount = $trainData ? $trainData.source.shape[0] : 0;

	const renderPreview = () => {
		tf.tidy(() => {
			const source = outputToSprite(getSample($trainData.source, $inputShape, previewIndex), [$spriteHeight, $spriteWidth, $spriteChannels]);
			tf.browser.toPixels(source, sourcePreview);
			
			const target = outputToSprite(getSample($trainData.target, $inputShape, previewIndex), [$spriteHeight, $spriteWidth, $spriteChannels]);
			tf.browser.toPixels(target, targetPreview);
		});
	};

	const backPreview = () => {
		if (previewIndex - 1 >= 0) previewIndex -= 1;
		else previewIndex = spritePairCount - 1;
		renderPreview();
	};
	
	const nextPreview = () => {
		if (previewIndex + 1 < spritePairCount) previewIndex += 1;
		else previewIndex = 0;
		renderPreview();
	};

	onMount(() => {
		renderPreview();
	});
</script>

<div>
	<span>
		<button on:click={backPreview}>Back</button>
		<canvas bind:this={sourcePreview}></canvas>
		<canvas bind:this={targetPreview}></canvas>
		<button on:click={nextPreview}>Next</button>
	</span>
	<p>Sprite {previewIndex + 1}/{spritePairCount}</p>
</div>

<style>
	canvas {
		background-color: black;
		width: 64px;
		height: 64px;
	}

	div {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 1em;
		padding: 2em 1em 1em;
	}
	
	span {
		display: flex;
	}

	button {
		padding: 0.25em 0.5em;
		border-radius: 0;
		cursor: pointer;
	}
</style>
