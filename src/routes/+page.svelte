<script>
	import { imageWidth, imageHeight, imageChannels } from "$lib/stores/data.js";
	import { generator, discriminator, gan } from "$lib/stores/model.js";
	import { createGenerator, createDiscriminator, createGan } from "$lib/actions/model.js";

	import Data from "$lib/components/data/data.svelte";
	import Train from "$lib/components/train/train.svelte";
	import Generate from "$lib/components/generate.svelte";

	$generator = createGenerator([$imageWidth, $imageHeight, $imageChannels]);
	$discriminator = createDiscriminator([$imageWidth, $imageHeight, $imageChannels * 2]);
	$gan = createGan($generator, $discriminator);
</script>

<main>
	<Data />
	<Train />
	<Generate />
</main>

<style>
	main {
		margin-inline: auto;
		display: grid;
		align-content: start;
		gap: 2em;
		padding: 0 2em;
		/* breaks at 450px */
	}
</style>
