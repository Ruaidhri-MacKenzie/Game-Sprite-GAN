<script>
	import Linechart from "$lib/components/common/linechart.svelte";

	export let values;
	export let name;
</script>

<details>
	<summary>{`${name} Loss Chart`}</summary>
	<Linechart {values} name={`${name} Loss`} xLabel="Epoch" yLabel={`${name} Loss`} />
</details>

<style>
	summary {
		user-select: none;
		cursor: pointer;
	}
</style>
