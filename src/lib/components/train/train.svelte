<script>
	import * as tf from "@tensorflow/tfjs";
	import { trainData, testData, spriteWidth, spriteHeight, spriteChannels } from "$lib/stores/data.js";
	import { generator, discriminator, gan, inputShape, patchShape } from "$lib/stores/model.js";
	import { training, epochs, genLossHistory, discLossHistory, epochLog } from "$lib/stores/train.js";
	import { getBatch, outputToSprite, spriteToImage } from "$lib/utils/data.utils.js";
	import Linechart from "$lib/components/train/linechart.svelte";
	import EpochReport from "$lib/components/train/epoch-report.svelte";
	import EpochTest from "$lib/components/train/epoch-test.svelte";
	
	let epoch = 0;
	let step = 0;
	let stepsPerEpoch = 0;
	let batchSize = 1;
	let tensors = 0;
	let testSources = [];
	let testTargets = [];
	let epochTests = [];

	const trainModel = async (event) => {
		$training = true;
		epochLog.reset();

		stepsPerEpoch = $trainData.source.shape[0];
		let startTime = Date.now();
		const testInputs = [
			getBatch($trainData.source, $inputShape, 0, 1),
			getBatch($testData.source, $inputShape, 0, 1),
			getBatch($testData.source, $inputShape, 1, 1),
			getBatch($testData.source, $inputShape, 2, 1),
		];

		const testOutputs = [
			getBatch($trainData.target, $inputShape, 0, 1),
			getBatch($testData.target, $inputShape, 0, 1),
			getBatch($testData.target, $inputShape, 1, 1),
			getBatch($testData.target, $inputShape, 2, 1),
		];

		testSources = await Promise.all(testInputs.map(async (input) => {
			const sprite = outputToSprite(input, [$spriteHeight, $spriteWidth, $spriteChannels]);
			const image = await spriteToImage(sprite);
			sprite.dispose();
			return image;
		}));

		testTargets = await Promise.all(testOutputs.map(async (output) => {
			const sprite = outputToSprite(output, [$spriteHeight, $spriteWidth, $spriteChannels]);
			const image = await spriteToImage(sprite);
			sprite.dispose();
			return image;
		}));

		// Generate outputs for test data
		const testGenOutputs = await Promise.all(testInputs.map(testInput => {
			return $generator.predict(testInput);
		}));

		// Convert tensors to images
		const testImages = await Promise.all(testGenOutputs.map(async (testOutput) => {
			const sprite = outputToSprite(testOutput, [$spriteHeight, $spriteWidth, $spriteChannels]);
			const image = await spriteToImage(sprite);
			tf.dispose([testOutput, sprite]);
			return image;
		}));

		// Update test images list
		epochTests = [testImages];

		// Train the models
		for (let i = 0; i < $epochs; i++) {
			epoch = i;
			tensors = tf.memory().numTensors;
			startTime = Date.now();

			for (let j = 0; j < stepsPerEpoch; j++) {
				step = j;

				const realLabel = tf.ones([batchSize, ...$patchShape]);
				const fakeLabel = tf.zeros([batchSize, ...$patchShape]);
				const realInput = getBatch($trainData.source, $inputShape, step, batchSize);
				const realOutput = getBatch($trainData.target, $inputShape, step, batchSize);
				const fakeOutput = await $generator.predict(realInput);

				// Train the discriminator
				const [realLoss, fakeLoss] = await Promise.all([
					$discriminator.trainOnBatch([realInput, realOutput], realLabel),
					$discriminator.trainOnBatch([realInput, fakeOutput], fakeLabel),
				]);
				const discriminatorLoss = realLoss + fakeLoss;

				// Train the generator
				$discriminator.trainable = false;
				const generatorLoss = await $gan.trainOnBatch(realInput, realLabel);
				$discriminator.trainable = true;
				
				// Epoch Report
				if (step === stepsPerEpoch - 1) {
					const epochTime = ((Date.now() - startTime) / 1000);
					const epochReport = `${epoch + 1}/${$epochs}: ${epochTime.toFixed(2)}sec - Discriminator loss: ${discriminatorLoss.toFixed(4)}, Generator loss: ${generatorLoss.toFixed(4)}`;
					console.log(epochReport);
					epochLog.addEntry(epochReport);
					genLossHistory.addLoss(epoch + 1, generatorLoss);
					discLossHistory.addLoss(epoch + 1, discriminatorLoss);
					
					// Generate outputs for test data
					const testOutputs = await Promise.all(testInputs.map(testInput => {
						return $generator.predict(testInput);
					}));

					// Convert tensors to images
					const testImages = await Promise.all(testOutputs.map(async (testOutput) => {
						const sprite = outputToSprite(testOutput, [$spriteHeight, $spriteWidth, $spriteChannels]);
						const image = await spriteToImage(sprite);
						tf.dispose([testOutput, sprite]);
						return image;
					}));

					// Update test images list
					epochTests = [testImages, ...epochTests];
				}
				
				tf.dispose([realInput, realOutput, fakeOutput, realLabel, fakeLabel]);
			}
		}

		epochLog.addEntry("Training complete");
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

	<Linechart values={$genLossHistory} name="Generator Loss" xLabel="Epoch" yLabel="Generator Loss" />
	<Linechart values={$discLossHistory} name="Discriminator Loss" xLabel="Epoch" yLabel="Discriminator Loss" />
	<EpochReport log={$epochLog} step={step} steps={stepsPerEpoch} />
	<EpochTest sources={testSources} targets={testTargets} tests={epochTests} />
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
