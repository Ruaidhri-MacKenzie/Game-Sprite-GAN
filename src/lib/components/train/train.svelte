<script>
	import * as tf from "@tensorflow/tfjs";
	import { dataset, imageWidth, imageHeight, imageChannels } from "$lib/stores/data.js";
	import { generator, discriminator, gan } from "$lib/stores/model.js";
	
	import Linechart from "$lib/components/train/linechart.svelte";

	let epochs = 20;
	let batchSize = 8;
	$: batchesPerEpoch = $dataset ? Math.floor($dataset.source.shape[0] / batchSize) : 0;

	let epoch = 0;
	let batch = 0;

	let discRealAccuracy = 0;
	let discFakeAccuracy = 0;
	// let genLossHistory = Array(100).fill(0).map(y => Math.random() * 100 + 50).map((y, x) => ({ x, y, }));
	let genLossHistory = [];
	let discLossHistory = [];

	let training = false;

	const selectBatch = (dataset, index = 0, batchSize = 1) => {
		// Select batch of samples for training
		const source = dataset.source.slice([index, 0, 0, 0], [batchSize, $imageHeight, $imageWidth, $imageChannels]);
		const target = dataset.target.slice([index, 0, 0, 0], [batchSize, $imageHeight, $imageWidth, $imageChannels]);
		return [source, target];
	};

	const createDiscriminatorData = async (generator, source, target, batchSize = 2) => {
		const halfBatch = Math.floor(batchSize / 2);

		// Select first half of the batch as real source/target inputs
		const realSource = source.slice([0, 0, 0, 0], [halfBatch, $imageHeight, $imageWidth, $imageChannels]);
		const realTarget = target.slice([0, 0, 0, 0], [halfBatch, $imageHeight, $imageWidth, $imageChannels]);
		const realFeatures = tf.concat([realSource, realTarget], 3);
		const realLabels = tf.ones([halfBatch, 1]);

		// Select second half of the batch as fake source and generate fake targets
		const fakeSource = source.slice([halfBatch, 0, 0, 0], [halfBatch, $imageHeight, $imageWidth, $imageChannels]);
		const fakeTarget = await generator.predict(fakeSource);
		const fakeFeatures = tf.concat([fakeSource, fakeTarget], 3);
		const fakeLabels = tf.zeros([halfBatch, 1]);

		// Merge real and fake samples to create training set for the discriminator
		const features = tf.concat([realFeatures, fakeFeatures], 0);
		const labels = tf.concat([realLabels, fakeLabels], 0);
		return [features, labels];
	};

	const displayBatchReport = async (epoch, batch, batchesPerEpoch, discLoss, genLoss) => {
		// Display loss and training accuracy
		genLossHistory = [...genLossHistory, genLoss];
		discLossHistory = [...discLossHistory, discLoss];
	};

	const displayEpochReport = async (generator, source, batchSize) => {
		// Display info at end of epoch
	};

	const trainModel = async (event) => {
		training = true;

		for (epoch = 0; epoch < epochs; epoch++) {
			for (batch = 0; batch < batchesPerEpoch; batch++) {
				// Select batch of real samples (real source images and their real target images)
				const [source, target] = selectBatch($dataset, batch, batchSize);
				
				// Create dataset of combined source/target images (concat on imageChannel) and real/fake classification (0 or 1)
				const [features, labels] = await createDiscriminatorData($generator, source, target, batchSize);
				console.log(features.shape);
				console.log(labels.shape);

				// Train the discriminator, then freeze weights while generator trains
				$discriminator.trainable = true;
				const [discLoss, discAcc] = await $discriminator.trainOnBatch(features, labels);
				$discriminator.trainable = false;
				
				// Train the generator through the combined GAN model
				const genLoss = await gan.trainOnBatch(source, target);

				// Display training info for this batch
				displayBatchReport(epoch, batch, batchesPerEpoch, discLoss, genLoss);
			}
			// Display training info for this epoch
			displayEpochReport(epoch, $generator, $discriminator, $dataset);
		}

		training = false;
	};
</script>

<section>
	<h2>Train</h2>
	{#if training}
		<p>Training...</p>
	{:else}
		<button disabled={!$dataset} on:click={trainModel}>Train</button>
	{/if}

	<p>Epoch: {epoch + 1}/{epochs} | Batch: {batch + 1}/{batchesPerEpoch}</p>

	<p>Discriminator Accuracy: Real {(discRealAccuracy * 100).toFixed(2)}% | Fake {(discFakeAccuracy * 100).toFixed(2)}%</p>

	{#if genLossHistory.length > 0}
		<Linechart values={genLossHistory} name="Generator Loss" xLabel="Batch" yLabel="Generator Loss"/>
	{/if}

	{#if discLossHistory.length > 0}
		<Linechart values={discLossHistory} name="Discriminator Loss" xLabel="Batch" yLabel="Discriminator Loss"/>
	{/if}

	<!-- Epoch Report - Display source/generated images -->
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
		cursor: pointer;
	}
</style>
