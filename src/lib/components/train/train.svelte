<script>
	import * as tf from "@tensorflow/tfjs";
	import { dataset, imageWidth, imageHeight, imageChannels } from "$lib/stores/data.js";
	import { generator, discriminator } from "$lib/stores/model.js";
	import { training, epochs, genLossHistory, discLossHistory } from "$lib/stores/train.js";
	import Linechart from "$lib/components/train/linechart.svelte";
	
	let epoch = 0;
	let step = 0;
	let stepsPerEpoch = 0;
	let batchSize = 1;

	const trainModel = async (event) => {
		$training = true;
		console.log("Training...");

		stepsPerEpoch = $dataset.source.shape[0];

		// Train the models
		for (let i = 0; i < $epochs; i++) {
			epoch = i;
			console.log(`Epoch ${epoch + 1}/${$epochs}`);
			for (let j = 0; j < stepsPerEpoch; j++) {
				step = j;

				// Train the discriminator
				const realInput = $dataset.source.slice([step, 0, 0, 0], [1, $imageHeight, $imageWidth, $imageChannels]);
				const realOutput = $dataset.target.slice([step, 0, 0, 0], [1, $imageHeight, $imageWidth, $imageChannels]);
				const fakeOutput = await $generator.predict(realInput);
				const realLoss = await $discriminator.trainOnBatch([realInput, realOutput], tf.ones([batchSize, 15, 15, 1]));
				const fakeLoss = await $discriminator.trainOnBatch([realInput, fakeOutput], tf.zeros([batchSize, 15, 15, 1]));
				const discriminatorLoss = realLoss + fakeLoss;

				// Train the generator
				const generatorLoss = await $generator.trainOnBatch(realInput, realOutput, tf.ones([batchSize, 15, 15, 1]));
				console.log(`Epoch: ${epoch + 1} - ${step + 1}/${stepsPerEpoch} - Discriminator loss: ${discriminatorLoss}, Generator loss: ${generatorLoss}`);
				genLossHistory.update(current => [...current, { x: (epoch * stepsPerEpoch) + step, y: generatorLoss}])
				discLossHistory.update(current => [...current, { x: (epoch * stepsPerEpoch) + step, y: discriminatorLoss}])
			}
		}

		console.log("Training complete");
		$training = false;
	};
</script>

<section>
	<h2>Train</h2>
	{#if $training}
		<p>Training...</p>
		<p>Epoch: {epoch + 1}/{$epochs}, Step: {step + 1}/{stepsPerEpoch}</p>
	{:else}
		<button disabled={!$dataset} on:click={trainModel}>Train</button>
	{/if}

	{#if $genLossHistory && $genLossHistory.length}
		<Linechart values={$genLossHistory} name="Generator Loss" xLabel="Step" yLabel="Generator Loss"/>
	{/if}

	{#if $discLossHistory && $discLossHistory.length}
		<Linechart values={$discLossHistory} name="Discriminator Loss" xLabel="Step" yLabel="Discriminator Loss"/>
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
