<script>
	import * as tf from "@tensorflow/tfjs";
	import { inputShape, generator, discriminator, gan, genLearningRate, discLearningRate } from "$lib/stores/model.js";
	import { createGenerator, createDiscriminator, createGAN, downloadModel } from "$lib/utils/model.utils.js";
	import ModelPreview from "./model-preview.svelte";
	import Section from "$lib/components/common/section.svelte";

	const createModel = (event) => {
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
	};

	const saveModels = (event) => {
		if ($generator) downloadModel($generator, "generator");
		if ($discriminator) downloadModel($discriminator, "discriminator");
	};
</script>

<Section title="Model">
	<div>
		<button on:click={createModel}>Create Models</button>

		{#if $generator && $discriminator}
			<button on:click={saveModels}>Save Models</button>
		{/if}
		
		<ModelPreview model={$generator} name="Generator" />
		<ModelPreview model={$discriminator} name="Discriminator" />
		<ModelPreview model={$gan} name="GAN" />
	</div>
</Section>

<style>
	div {
		padding: 1em;
	}
</style>
