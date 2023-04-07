<script>
	import * as tfvis from "@tensorflow/tfjs-vis";

	export let values;
	export let name;
	export let xLabel;
	export let yLabel;
	export let xType = "ordinal";
	export let yType = "quantitative";
	export let zoomToFit = true;
	export let width = 600;
	export let height = 300;
	export let fontSize = 12;

	let chart;

	$: if (chart && values) {
		tfvis.render.linechart(chart, { values, series: [name] }, {
			xLabel, yLabel,
			xType, yType,
			width, height,
			zoomToFit,
			fontSize,
		});
	}
</script>

<div bind:this={chart} />

<style>
	div {
		padding: 1em;
		box-shadow: rgba(0, 0, 0, 0.12) 0px 1px 3px, rgba(0, 0, 0, 0.24) 0px 1px 2px;
	}
</style>
