<script>
	import * as tf from "@tensorflow/tfjs";
	import { trainData, testData } from "$lib/stores/data.js";
	import { generator, discriminator } from "$lib/stores/model.js";
	import { training, epochs, genLossHistory, discLossHistory } from "$lib/stores/train.js";
	import Linechart from "$lib/components/train/linechart.svelte";
	
	let inputShape = [32, 32, 4];
	let patchShape = [15, 15, 1];
	let epoch = 0;
	let step = 0;
	let stepsPerEpoch = 0;
	let batchSize = 1;
	let tensors = 0;

	const trainGenerator = async (genOutput, genLabel) => {
		const loss = await $generator.trainOnBatch(genOutput, genLabel);
		return loss;
	};

	const trainDiscriminator = async (input, labels) => {
		const loss = await $discriminator.trainOnBatch(input, labels);
		return loss;
	};

	const generateImage = async (source) => {
		const target = await $generator.predict(source);
		return target;
	};

	const trainModel = async (event) => {
		$training = true;
		console.log("Training...");

		stepsPerEpoch = $trainData.source.shape[0];
		let start = Date.now();

		// Train the models
		for (let i = 0; i < $epochs; i++) {
			epoch = i;
			tensors = tf.memory().numTensors;
			start = Date.now();

			for (let j = 0; j < stepsPerEpoch; j++) {
				step = j;

				const realLabel = tf.ones([batchSize, ...patchShape]);
				const fakeLabel = tf.zeros([batchSize, ...patchShape]);
				const realInput = $trainData.source.slice([step, 0, 0, 0], [1, ...inputShape]);
				const realOutput = $trainData.target.slice([step, 0, 0, 0], [1, ...inputShape]);
				const fakeOutput = await generateImage(realInput);

				// Train the discriminator
				const realLoss = await trainDiscriminator([realInput, realOutput], realLabel);
				const fakeLoss = await trainDiscriminator([realInput, fakeOutput], fakeLabel);
				const discriminatorLoss = realLoss + fakeLoss;

				// Train the generator
				const generatorLoss = await trainGenerator(realInput, realOutput);

				// Epoch Report
				if (step === stepsPerEpoch - 1) {
					console.log(`Epoch: ${epoch + 1}/${$epochs} - Time taken: ${((Date.now() - start) / 1000).toFixed(2)}sec - Discriminator loss: ${discriminatorLoss.toFixed(4)}, Generator loss: ${generatorLoss.toFixed(4)}`);
					// genLossHistory.update(current => [...current, { x: epoch, y: generatorLoss}]);
					// discLossHistory.update(current => [...current, { x: epoch, y: discriminatorLoss}]);
				}
				
				tf.dispose([realInput, realOutput, fakeOutput, realLabel, fakeLabel]);
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
		<p>Active Tensors: {tensors}</p>
		<p>Epoch: {epoch + 1}/{$epochs}, Step: {step + 1}/{stepsPerEpoch}</p>
	{:else}
		<label>
			<p>Epochs:</p>
			<input bind:value={$epochs} type="number" min={1} />
		</label>
		<button disabled={!$trainData} on:click={trainModel}>Train</button>
	{/if}

	{#if $genLossHistory && $genLossHistory.length}
		<Linechart values={$genLossHistory} name="Generator Loss" xLabel="Epoch" yLabel="Generator Loss" />
	{/if}

	{#if $discLossHistory && $discLossHistory.length}
		<Linechart values={$discLossHistory} name="Discriminator Loss" xLabel="Epoch" yLabel="Discriminator Loss" />
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

	label {
		display: flex;
		align-items: center;
		gap: 1em;
	}

	input {
		width: 7ch;
	}
</style>
