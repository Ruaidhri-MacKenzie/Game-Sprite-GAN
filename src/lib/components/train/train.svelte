<script>
	import { dataset } from "$lib/stores/data.js";
	import { worker, training, step } from "$lib/stores/train.js";
	import Linechart from "$lib/components/train/linechart.svelte";
	
	export let steps = 1000;
	export let genLossHistory;
	export let discLossHistory;

	const train = async (event) => {
		$training = true;
		$worker.postMessage({ sources: $dataset.source.dataSync(), targets: $dataset.target.dataSync(), steps });
	};
</script>

<section>
	<h2>Train</h2>
	{#if $training}
		<p>Training...</p>
		<p>Step: {$step}/{steps}</p>
	{:else}
		<button disabled={!$dataset} on:click={train}>Train</button>
	{/if}

	{#if genLossHistory && genLossHistory.length}
		<Linechart values={genLossHistory} name="Generator Loss" xLabel="Step" yLabel="Generator Loss"/>
	{/if}

	{#if discLossHistory && discLossHistory.length}
		<Linechart values={discLossHistory} name="Discriminator Loss" xLabel="Step" yLabel="Discriminator Loss"/>
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
</style>
