<script>
	import Data from "$lib/components/data/data.svelte";
	import Model from "$lib/components/model.svelte";
	import Train from "$lib/components/train/train.svelte";
	import Generate from "$lib/components/generate.svelte";
</script>

<main>
	<Data />
	<Model />
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
