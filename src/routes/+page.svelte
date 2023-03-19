<script>
	import { onMount } from "svelte";
	import Data from "$lib/components/data/data.svelte";
	import Train from "$lib/components/train/train.svelte";
	import Generate from "$lib/components/generate.svelte";
	import { worker, training, step, genLossHistory, discLossHistory } from "$lib/stores/train.js";

	let steps = 5000;

	let target = null;
	
	onMount(() => {
		$worker = new Worker(new URL("./worker.js", import.meta.url));
		$worker.addEventListener("message", (event) => {
			if (event.data.error) {
				$training = false;
				console.log(event.data.error);
			}
			else if (event.data.success) {
				$training = false;
				console.log("Training complete");
			}
			else if (event.data.step) {
				$step = event.data.step;
				$genLossHistory.push({ x: $step, y: event.data.genLoss });
				$discLossHistory.push({ x: $step, y: event.data.discLoss });
			}
			else if (event.data.target) {
				target = event.data.target;			
			}
		});
	});
</script>

<main>
	<Data />
	<Train {steps} genLossHistory={$genLossHistory} discLossHistory={$discLossHistory} />
	<Generate bind:target />
</main>

<style>
	main {
		margin-inline: auto;
		display: grid;
		align-content: start;
		gap: 2em;
		padding: 0 2em;
		/* breaks at 450px */
	}
</style>
