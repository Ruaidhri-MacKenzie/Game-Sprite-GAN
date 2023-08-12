<script>
	import { beforeUpdate, afterUpdate } from "svelte";
	import { epochLog, step, stepsPerEpoch } from "$lib/stores/train.js";
	
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
	<progress value={$step} max={$stepsPerEpoch}></progress>
	<ul bind:this={list}>
		{#each $epochLog as report}
			<li>{report}</li>
		{/each}
	</ul>
</div>

<style>
	div {
		display: flex;
		flex-direction: column;
	}

	h3 {
		text-align: center;
	}
	
	ul {
		height: 9em;
		overflow-y: scroll;
		list-style-type: none;
		border: 1px solid black;
		background-color: hsl(0 0% 98%);
	}

	li {
		padding: 0.5em;
		background-color: hsl(0 0% 98%);
	}

	li:nth-child(2n) {
		background-color: hsl(0 0% 95%);
	}
</style>
