<script>
	import { beforeUpdate, afterUpdate } from "svelte";
	
	export let log = [];
	export let step = 0;
	export let steps = 0;

	let list;
	let autoScroll = true;

	beforeUpdate(() => {
		if (list) autoScroll = (list.scrollTop + list.clientHeight >= list.scrollHeight);
	});

	afterUpdate(() => {
		if (list && autoScroll) list.scrollTop = list.scrollHeight;
	});
</script>
<div>
	<h3>Epoch Report</h3>
	<progress value={step} max={steps}></progress>
	<ul bind:this={list}>
		{#each log as report}
			<li>{report}</li>
		{/each}
	</ul>
</div>

<style>
	ul {
		height: 9em;
		overflow-y: scroll;
		list-style-type: none;
	}

	li {
		padding: 0.5em;
		background-color: hsl(0 0% 98%);
	}

	li:nth-child(2n) {
		background-color: hsl(0 0% 95%);
	}
</style>
