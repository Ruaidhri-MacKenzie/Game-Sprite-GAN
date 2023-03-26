<script>
	import { dataset } from "$lib/stores/data.js";
	
	import Upload from "./upload.svelte";
	import Preview from "./preview.svelte";
</script>

<section>
	<h2>Data</h2>
	<div>
		<Upload />
		{#if $dataset}
			<Preview />
		{/if}
	</div>
</section>

<style>
	section {
		padding: 1em;
		box-shadow: 0 3px 8px hsl(0 0% 0% / 0.24);
	}

	div {
		display: flex;
		flex-direction: column;
	}
</style>
