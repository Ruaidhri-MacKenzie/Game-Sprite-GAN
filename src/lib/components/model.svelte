<script>
	import * as tf from "@tensorflow/tfjs";
	import * as tfvis from "@tensorflow/tfjs-vis";
	import { inputShape, generator, discriminator, gan, genLearningRate, discLearningRate } from "$lib/stores/model.js";
	import { createGenerator, createDiscriminator, createGAN, downloadModel } from "$lib/utils/model.utils.js";

	let genSummary;
	let discSummary;
	let ganSummary;
	let loading = false;
	$: status = (loading) ? "loading..." : ($generator && $discriminator) ? "✓" : "X";

	const createModel = (event) => {
		loading = true;
		
		$generator = createGenerator($inputShape);
		$generator.compile({ optimizer: tf.train.adam($genLearningRate), loss: "meanAbsoluteError" });

		$discriminator = createDiscriminator($inputShape);
		$discriminator.compile({ optimizer: tf.train.adam($discLearningRate), loss: "binaryCrossentropy" });
		
		$gan = createGAN($inputShape, $generator, $discriminator);
		$gan.compile({ optimizer: tf.train.adam($genLearningRate), loss: "binaryCrossentropy" });

		// Attempt at adding L1 loss
		// $gan = tf.model({ inputs: inputSource, outputs: [discOut, genOut] });
		// $gan.compile({ optimizer: tf.train.adam($genLearningRate), loss: ["binaryCrossentropy", "meanAbsoluteError"] });
		// const generatorLoss = await $gan.trainOnBatch(realInput, [realLabel, realOutput]);

		tfvis.show.modelSummary(genSummary, $generator);
		tfvis.show.modelSummary(discSummary, $discriminator);
		tfvis.show.modelSummary(ganSummary, $gan);

		loading = false;
	};

	const saveModels = (event) => {
		if ($generator) downloadModel($generator, "generator");
		if ($discriminator) downloadModel($discriminator, "discriminator");
	};
</script>

<section>
	<h2>Model</h2>
	<button on:click={createModel}>Create Models</button>
	{#if $generator && $discriminator}
		<button on:click={saveModels}>Save Models</button>
	{/if}
	<details bind:this={genSummary}>
		<summary>Generator: <span class:active={$generator}>{status}</span></summary>
	</details>
	<details bind:this={discSummary}>
		<summary>Discriminator: <span class:active={$discriminator}>{status}</span></summary>
	</details>
	<details bind:this={ganSummary}>
		<summary>GAN: <span class:active={$gan}>{status}</span></summary>
	</details>
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
		padding: 0.25em 0.5em;
	}

	span {
		color: red;
	}

	span.active {
		color: green;
	}
</style>
