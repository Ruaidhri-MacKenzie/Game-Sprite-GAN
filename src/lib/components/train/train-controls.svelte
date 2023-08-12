<script>
	import * as tf from "@tensorflow/tfjs";
	import { trainData, testInputs, spriteWidth, spriteHeight, spriteChannels } from "$lib/stores/data.js";
	import { generator, discriminator, gan, inputShape } from "$lib/stores/model.js";
	import { training, step, stepsPerEpoch, epochLog, epochTests, genLossHistory, discLossHistory } from "$lib/stores/train.js";
	import { getBatch, outputsToImages } from "$lib/utils/data.utils.js";
	import { epochReportToString } from "$lib/utils/train.utils.js";
	
	let epoch = 0;
	let epochs = 50;
	let batchSize = 1;
	let tensors = 0;

	$: patchShape = [Math.floor($inputShape[0] / 2), Math.floor($inputShape[1] / 2), 1];

	const generateFromList = async (list) => {
		return await Promise.all(list.map(async (input) => {
			return await $generator.predict(input);
		}));
	};

	const trainModel = async (event) => {
		$training = true;
		epochLog.reset();
		$stepsPerEpoch = $trainData.source.shape[0];
		let startTime = Date.now();
		
		// Generate outputs for test data
		const testGenOutputs = await generateFromList($testInputs);
		
		// Convert generated outputs to images
		const spriteShape = [$spriteHeight, $spriteWidth, $spriteChannels];
		const testImages = await outputsToImages(testGenOutputs, spriteShape);

		// Update test images list
		$epochTests = [testImages];

		const realLabel = tf.ones([batchSize, ...patchShape]);
		const fakeLabel = tf.zeros([batchSize, ...patchShape]);

		// Train the models
		for (let i = 0; i < epochs; i++) {
			epoch = i;
			tensors = tf.memory().numTensors;
			startTime = Date.now();

			for (let j = 0; j < $stepsPerEpoch; j++) {
				$step = j;

				// Prepare training data
				const realInput = getBatch($trainData.source, $inputShape, $step, batchSize);
				const realOutput = getBatch($trainData.target, $inputShape, $step, batchSize);
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
				if ($step === $stepsPerEpoch - 1) {
					// Epoch Report
					const epochReport = {
						epoch: epoch + 1,
						epochs,
						time: ((Date.now() - startTime) / 1000),
						generatorLoss,
						discriminatorLoss,
					};
					epochLog.addEntry(epochReportToString(epochReport));

					// Loss History
					genLossHistory.addLoss(epoch + 1, generatorLoss);
					discLossHistory.addLoss(epoch + 1, discriminatorLoss);
					
					// Generate test images
					const epochTestOutputs = await generateFromList($testInputs);
					const testImages = await outputsToImages(epochTestOutputs, spriteShape);
					epochTests.addTest(testImages);
					tf.dispose([epochTestOutputs, testImages]);
				}
				
				tf.dispose([realInput, realOutput, fakeOutput]);
			}
		}

		tf.dispose([testGenOutputs, realLabel, fakeLabel]);
		epochLog.addEntry("Training complete");
		$training = false;
	};
</script>

<div>
	{#if $training}
		<p>Training...</p>
		<p>Batch Size: {batchSize}</p>
		<p>Active Tensors: {tensors}</p>
		<p>Epoch: {epoch + 1}/{epochs}, Step: {$step + 1}/{$stepsPerEpoch}</p>
	{:else}
		<form>
			<label for="epochs">Epochs:</label>
			<input bind:value={epochs} id="epochs" type="number" min={1} />
			
			<label for="batch-size">Batch Size:</label>
			<input bind:value={batchSize} id="batch-size" type="number" min={1} />
		</form>
		
		<button disabled={!$trainData || !$generator || !$discriminator || !$gan} on:click={trainModel}>Train</button>
	{/if}
</div>

<style>
	div {
		display: grid;
		justify-items: center;
		gap: 1em;
		padding: 1em;
	}

	form {
		width: fit-content;
		display: grid;
		grid-template-columns: auto 1fr;
		align-items: center;
		gap: 0.5em;
	}

	label {
		text-align: right;
	}

	input {
		width: 7ch;
	}
</style>
