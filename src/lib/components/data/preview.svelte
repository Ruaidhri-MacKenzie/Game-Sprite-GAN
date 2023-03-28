<script>
	import * as tf from "@tensorflow/tfjs";
	import { trainData, spriteWidth, spriteHeight, spriteChannels } from "$lib/stores/data.js";
	import { onMount } from "svelte";

	let sourcePreview;
	let targetPreview;
	let previewIndex = 0;
	$: spritePairCount = $trainData ? $trainData.source.shape[0] : 0;

	const getSample = (dataset, index) => {
		return tf.tidy(() => {
			let sample = dataset.slice([index, 0, 0, 0], [1, $spriteHeight, $spriteWidth, $spriteChannels]);
			sample = sample.reshape([$spriteHeight, $spriteWidth, $spriteChannels]);
			sample = sample.add(1).div(2);
			return sample;
		});
	};

	const renderPreview = () => {
		tf.tidy(() => {
			const source = getSample($trainData.source, previewIndex);
			tf.browser.toPixels(source, sourcePreview);
			
			const target = getSample($trainData.target, previewIndex);
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
