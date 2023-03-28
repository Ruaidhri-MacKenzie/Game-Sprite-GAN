<script>
	import * as tf from "@tensorflow/tfjs";
	import { trainData, testData, imageWidth, imageHeight, imageChannels } from "$lib/stores/data.js";
	import { generator, discriminator } from "$lib/stores/model.js";
	import { training, epochs, genLossHistory, discLossHistory } from "$lib/stores/train.js";
	import Linechart from "$lib/components/train/linechart.svelte";
	
	let epoch = 0;
	let step = 0;
	let stepsPerEpoch = 0;
	let batchSize = 1;
	let tensors = 0;

	const trainModel = async (event) => {
		$training = true;
		console.log("Training...");

		stepsPerEpoch = $trainData.source.shape[0];
		let start = Date.now();

		// Train the models
		for (let i = 0; i < $epochs; i++) {
			epoch = i;
			console.log(`Epoch ${epoch + 1}/${$epochs} - Time taken: ${((Date.now() - start) / 1000).toFixed(2)}sec`);
			start = Date.now();

			for (let j = 0; j < stepsPerEpoch; j++) {
				step = j;
				tensors = tf.memory().numTensors;

				const realInput = $trainData.source.slice([step, 0, 0, 0], [1, $imageHeight, $imageWidth, $imageChannels]);
				const realOutput = $trainData.target.slice([step, 0, 0, 0], [1, $imageHeight, $imageWidth, $imageChannels]);
				const fakeOutput = await $generator.predict(realInput);
				
				// Train the discriminator
				const onesLabel = tf.ones([batchSize, 15, 15, 1]);
				const zerosLabel = tf.zeros([batchSize, 15, 15, 1]);
				const realLoss = await $discriminator.trainOnBatch([realInput, realOutput], onesLabel);
				const fakeLoss = await $discriminator.trainOnBatch([realInput, fakeOutput], zerosLabel);
				const discriminatorLoss = realLoss + fakeLoss;

				// Train the generator
				const generatorLoss = await $generator.trainOnBatch(realInput, realOutput);
				// const generatorLoss = await $generator.trainOnBatch(fakeOutput, tf.onesLike(fakeOutput));

				console.log(`Epoch: ${epoch + 1}/${$epochs} - ${step + 1}/${stepsPerEpoch} - Discriminator loss: ${discriminatorLoss.toFixed(4)}, Generator loss: ${generatorLoss.toFixed(4)}`);
				genLossHistory.update(current => [...current, { x: (epoch * stepsPerEpoch) + step, y: generatorLoss}]);
				discLossHistory.update(current => [...current, { x: (epoch * stepsPerEpoch) + step, y: discriminatorLoss}]);

				realInput.dispose();
				realOutput.dispose();
				fakeOutput.dispose();
				onesLabel.dispose();
				zerosLabel.dispose();
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
		<p>Tensors: {tensors}</p>
		<p>Epoch: {epoch + 1}/{$epochs}, Step: {step + 1}/{stepsPerEpoch}</p>
	{:else}
		<button disabled={!$trainData} on:click={trainModel}>Train</button>
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
