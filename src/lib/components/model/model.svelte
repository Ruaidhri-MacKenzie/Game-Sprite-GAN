<script>
	import * as tf from "@tensorflow/tfjs";
	import { inputShape, generator, discriminator, gan } from "$lib/stores/model.js";
	import { createGenerator, createDiscriminator, createGAN, downloadModel } from "$lib/utils/model.utils.js";
	import ModelPreview from "./model-preview.svelte";
	import Section from "$lib/components/common/section.svelte";
	import Checkmark from "$lib/components/common/checkmark.svelte";

	let genLearningRate = 0.0002;
	let discLearningRate = 0.0002;

	$: generatorIsValid = Boolean($generator);
	$: discriminatorIsValid = Boolean($discriminator);
	$: ganIsValid = Boolean($gan);

	const createModel = (event) => {
		$generator = createGenerator($inputShape);
		$generator.compile({ optimizer: tf.train.adam(genLearningRate), loss: "meanAbsoluteError" });

		$discriminator = createDiscriminator($inputShape);
		$discriminator.compile({ optimizer: tf.train.adam(discLearningRate), loss: "binaryCrossentropy" });
		
		$gan = createGAN($inputShape, $generator, $discriminator);
		$gan.compile({ optimizer: tf.train.adam(genLearningRate), loss: "binaryCrossentropy" });

		// Attempt at adding L1 loss
		// $gan = tf.model({ inputs: inputSource, outputs: [discOut, genOut] });
		// $gan.compile({ optimizer: tf.train.adam(genLearningRate), loss: ["binaryCrossentropy", "meanAbsoluteError"] });
		// const generatorLoss = await $gan.trainOnBatch(realInput, [realLabel, realOutput]);
	};

	const saveGenerator = (event) => {
		if ($generator) downloadModel($generator, "generator");
	};

	const saveDiscriminator = (event) => {
		if ($discriminator) downloadModel($discriminator, "discriminator");
	};
</script>

<Section title="Model">
	<div>
		<button on:click={createModel}>Create Models</button>
		
		<span>
			<h3>Generator: <Checkmark checked={generatorIsValid} /></h3>
			{#if generatorIsValid}
				<button on:click={saveGenerator}>Save Generator</button>
				<ModelPreview model={$generator} name="Generator" />
			{/if}
		</span>
	
		<span>
			<h3>Discriminator: <Checkmark checked={discriminatorIsValid} /></h3>
			{#if discriminatorIsValid}
				<button on:click={saveDiscriminator}>Save Discriminator</button>
				<ModelPreview model={$discriminator} name="Discriminator" />
			{/if}
		</span>

		<span>
			<h3>GAN (Combined): <Checkmark checked={ganIsValid} /></h3>
			{#if ganIsValid}
				<ModelPreview model={$gan} name="GAN" />
			{/if}
		</span>
	</div>
</Section>

<style>
	div {
		display: grid;
		justify-items: center;
		gap: 1em;
		padding: 1em;
	}

	span {
		display: grid;
		justify-items: center;
	}

	h3 {
		padding: 0.25em;
	}
</style>
