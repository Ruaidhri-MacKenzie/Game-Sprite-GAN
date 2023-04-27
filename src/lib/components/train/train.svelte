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

	const epochReportToString = (report) => {
		return `${report.epoch}/${report.epochs}: ${report.time.toFixed(2)}sec | Generator loss: ${report.generatorLoss.toFixed(4)} - Discriminator loss: ${report.discriminatorLoss.toFixed(4)}`;
	};

	const printEpochReport = (report) => {
		console.log(epochReportToString(report));
	};

	const generateFromList = async (list) => {
		return await Promise.all(list.map(async (input) => {
			return await $generator.predict(input);
		}));
	};

	const outputsToImages = async (outputs, imageShape) => {
		return await Promise.all(outputs.map(async (output) => {
			const sprite = outputToSprite(output, imageShape);
			const image = await spriteToImage(sprite);
			sprite.dispose();
			return image;
		}));
	};

	const trainModel = async (event) => {
		$training = true;
		epochLog.reset();
		stepsPerEpoch = $trainData.source.shape[0];
		let startTime = Date.now();
		
		// Create list of source inputs for testing
		const testInputs = [
			getBatch($trainData.source, $inputShape, 0, 1),
			getBatch($testData.source, $inputShape, 0, 1),
			getBatch($testData.source, $inputShape, 1, 1),
			getBatch($testData.source, $inputShape, 2, 1),
		];

		// Create list of target outputs for testing
		const testOutputs = [
			getBatch($trainData.target, $inputShape, 0, 1),
			getBatch($testData.target, $inputShape, 0, 1),
			getBatch($testData.target, $inputShape, 1, 1),
			getBatch($testData.target, $inputShape, 2, 1),
		];

		// Generate outputs for test data
		const testGenOutputs = await generateFromList(testInputs);

		const spriteShape = [$spriteHeight, $spriteWidth, $spriteChannels];

		// Convert source inputs to images
		testSources = await outputsToImages(testInputs, spriteShape);

		// Convert target outputs to images
		testTargets = await outputsToImages(testOutputs, spriteShape);

		// Convert generated outputs to images
		const testImages = await outputsToImages(testGenOutputs, spriteShape);

		// Update test images list
		epochTests = [testImages];

		// Train the models
		for (let i = 0; i < $epochs; i++) {
			epoch = i;
			tensors = tf.memory().numTensors;
			startTime = Date.now();

			for (let j = 0; j < stepsPerEpoch; j++) {
				step = j;

				// Prepare training data
				const realLabel = tf.ones([batchSize, ...$patchShape]);
				const fakeLabel = tf.zeros([batchSize, ...$patchShape]);
				const realInput = getBatch($trainData.source, $inputShape, step, batchSize);
				const realOutput = getBatch($trainData.target, $inputShape, step, batchSize);
				const fakeOutput = await $generator.predict(realInput);

				// Train the discriminator
				// const [realLoss, fakeLoss] = await Promise.all([
				// 	$discriminator.trainOnBatch([realInput, realOutput], realLabel),
				// 	$discriminator.trainOnBatch([realInput, fakeOutput], fakeLabel),
				// ]);
				const realLoss = await $discriminator.trainOnBatch([realInput, realOutput], realLabel);
				const fakeLoss = await $discriminator.trainOnBatch([realInput, fakeOutput], fakeLabel);
				const discriminatorLoss = realLoss + fakeLoss;

				// Train the generator
				$discriminator.trainable = false;
				const [ganLoss, l1Loss] = await Promise.all([
					$gan.trainOnBatch(realInput, realLabel),
					$generator.trainOnBatch(fakeOutput, realOutput),
				]);
				// const ganLoss = await $gan.trainOnBatch(realInput, realLabel);
				// const l1Loss = await $generator.trainOnBatch(fakeOutput, realOutput);
				const generatorLoss = ganLoss + l1Loss;
				$discriminator.trainable = true;
				
				// End of Epoch
				if (step === stepsPerEpoch - 1) {
					// Epoch Report
					const epochReport = {
						epoch: epoch + 1,
						epochs: $epochs,
						time: ((Date.now() - startTime) / 1000),
						generatorLoss,
						discriminatorLoss,
					};
					printEpochReport(epochReport);
					epochLog.addEntry(epochReportToString(epochReport));

					// Loss History
					genLossHistory.addLoss(epoch + 1, generatorLoss);
					discLossHistory.addLoss(epoch + 1, discriminatorLoss);
					
					// Generate test images
					const epochTestOutputs = await generateFromList(testInputs);
					const testImages = await outputsToImages(epochTestOutputs, spriteShape);
					tf.dispose(epochTestOutputs);
					epochTests = [testImages, ...epochTests];
				}
				
				tf.dispose([realInput, realOutput, fakeOutput, realLabel, fakeLabel]);
			}
		}

		tf.dispose([testInputs, testOutputs, testGenOutputs]);
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
