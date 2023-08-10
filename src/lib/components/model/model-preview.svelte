<script>
	import * as tfvis from "@tensorflow/tfjs-vis";
	import { afterUpdate } from "svelte";

	export let model;
	export let name = "Model";
	
	let element;
	$: active = Boolean(model);
	$: status = (active) ? "✓" : "X";

	afterUpdate(() => {
		if (model) tfvis.show.modelSummary(element, model);
	});
</script>

<details bind:this={element}>
	<summary>{name}: <span class:active>{status}</span></summary>
</details>

<style>
	span {
		color: red;
	}

	span.active {
		color: green;
	}
</style>
