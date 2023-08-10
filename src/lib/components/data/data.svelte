<script>
	import { trainData } from "$lib/stores/data.js";
	import Upload from "./upload.svelte";
	import Process from "./process.svelte";
	import Preview from "./preview.svelte";
	import Section from "$lib/components/common/section.svelte";

	let spritesheet;
	let sprites;
</script>

<Section title="Data">
	<div>
		<Upload bind:spritesheet />

		{#if spritesheet}
			<Process bind:spritesheet bind:sprites />
		{/if}
		
		{#if $trainData}
			<Preview dataset={$trainData} />
		{/if}
	</div>
</Section>

<style>
	div {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 3em;
		padding: 1em;
	}
</style>
