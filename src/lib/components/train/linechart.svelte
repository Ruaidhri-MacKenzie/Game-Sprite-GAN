<script>
	import * as tfvis from "@tensorflow/tfjs-vis";
	import { afterUpdate } from "svelte";

	export let values;
	export let name;
	export let xLabel;
	export let yLabel;
	export let zoomToFit = true;
	export let width = 600;
	export let height = 300;
	export let fontSize = 12;
	let chart;

	afterUpdate(() => {
		if (!chart) return;
		tfvis.render.linechart(chart, { values, series: [name] }, {
			zoomToFit,
			xLabel,
			yLabel,
			width,
			height,
			fontSize,
		});
	});
</script>

<div bind:this={chart} />

<style>
	div {
		padding: 1em;
		box-shadow: rgba(0, 0, 0, 0.12) 0px 1px 3px, rgba(0, 0, 0, 0.24) 0px 1px 2px;
	}
</style>
