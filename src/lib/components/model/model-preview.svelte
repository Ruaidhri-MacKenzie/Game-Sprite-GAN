<script>
	import * as tfvis from "@tensorflow/tfjs-vis";
	import { afterUpdate } from "svelte";

	export let model;
	export let name = "Model";
	
	let element;

	afterUpdate(() => {
		if (model) tfvis.show.modelSummary(element, model);
	});
</script>

<details bind:this={element}>
	<summary>{name} Summary</summary>
</details>

<style>
	details {
		padding: 0.25em;
	}
	
	summary {
		text-align: center;
		user-select: none;
		cursor: pointer;
	}
</style>
