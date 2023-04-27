<script>
	import * as tf from "@tensorflow/tfjs";
	import { trainData } from "$lib/stores/data.js";
	import { generator, inputShape } from "$lib/stores/model.js";
	import { getBatch, imageToSprite, cropInput } from "$lib/utils/data.utils.js";
	import { training, epochLog } from "$lib/stores/train.js";

	let pretrainData;
	let stepsPerEpoch = 0;
	let step = 0;

	const loadPretrainData = async () => {
		for (let i = 23; i <= 31; i++) {
			const sprite = await imageToSprite(`/data/pixel-art/${i}.png`, $inputShape[2]);
			const columns = Math.floor(sprite.shape[1] / $inputShape[1]);
			const rows = Math.floor(sprite.shape[0] / $inputShape[0]);

			// Split image into patches that match the input shape
			let splitSprites = cropInput(sprite.expandDims(), [rows * $inputShape[0], columns * $inputShape[1], $inputShape[2]]);
			splitSprites = splitSprites.reshape([rows, $inputShape[0], columns, $inputShape[1], $inputShape[2]]);
			splitSprites = splitSprites.transpose([0, 2, 1, 3, 4]);
			splitSprites = splitSprites.reshape([columns * rows, ...$inputShape]);
			if (pretrainData) {
				pretrainData = pretrainData.concat(splitSprites);
			}
			else {
				pretrainData = splitSprites;
			}
		}
	};

	const trainGeneratorDegrade = async () => {
		$training = true;
		stepsPerEpoch = pretrainData.shape[0];
		for (let epoch = 0; epoch < 1; epoch++) {
			for (let i = 0; i < stepsPerEpoch; i++) {
				step = i;
				const source = getBatch(pretrainData, $inputShape, step, 1);
				const target = tf.randomNormal([1, ...$inputShape]);
				const loss = await $generator.trainOnBatch(source, target);
				tf.dispose([source, target]);
				console.log(`Pretrain Step ${step + 1}/${stepsPerEpoch} - Loss: ${loss.toFixed(4)}`);
				if (step === stepsPerEpoch - 1) {
					epochLog.addEntry(`Pretrain Epoch ${epoch + 1} - Loss: ${loss.toFixed(4)}`);
					console.log(`Pretrain Epoch ${epoch + 1} - Loss: ${loss.toFixed(4)}`);
				}
			}
		}
		epochLog.addEntry("Pre-train Degrade complete");
		$training = false;
	};

	const trainGeneratorVAE = async () => {
		$training = true;
		stepsPerEpoch = $trainData.source.shape[0];
		for (let epoch = 0; epoch < 10; epoch++) {
			for (let i = 0; i < stepsPerEpoch; i++) {
				step = i;
				const source = getBatch($trainData.source, $inputShape, step, 1);
				const target = getBatch($trainData.target, $inputShape, step, 1);
				const loss = await $generator.trainOnBatch(source, target);
				tf.dispose([source, target]);
				if (step === stepsPerEpoch - 1) {
					console.log(`Epoch ${epoch + 1} - Loss: ${loss.toFixed(4)}`);
				}
			}
		}
		console.log("Pre-train VAE done");
		$training = false;
	};
</script>

<section>
	<h2>Pre-Train</h2>
	{#if $training}
		<p>Training...</p>
		<progress value={step} max={stepsPerEpoch}></progress>
	{:else}
		{#if !pretrainData}
			<button on:click={loadPretrainData} disabled={$training}>Load</button>
		{/if}
		<button on:click={trainGeneratorVAE} disabled={!$generator || $training}>Pre-Train VAE</button>
		<button on:click={trainGeneratorDegrade} disabled={!$generator || $training || !pretrainData}>Pre-Train Degrade</button>
	{/if}
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
		padding: 0.5em 1em;
	}
</style>
