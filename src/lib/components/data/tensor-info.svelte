<script>
	import { getMinValue, getMaxValue } from "$lib/utils/data.utils.js";

	export let tensor;
	
	$: shape = (tensor) ? tensor.shape : "";
	$: range = (tensor) ? `${getMinValue(tensor)}, ${getMaxValue(tensor)}` : "";
</script>

<div>
	<p>Shape:</p><p>[{shape}]</p>
	<p>Range:</p><p>[{range}]</p>
</div>

<style>
	div {
		width: fit-content;
		display: grid;
		grid-template-columns: auto 1fr;
		justify-items: start;
		align-items: center;
		gap: 0.5em;
	}
</style>
