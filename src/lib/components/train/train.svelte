<script>
	import { genLossHistory, discLossHistory } from "$lib/stores/train.js";
	import Section from "$lib/components/common/section.svelte";
	import TrainControls from "./train-controls.svelte";
	import EpochReport from "./epoch-report.svelte";
	import LossChart from "./loss-chart.svelte";
</script>

<Section title="Train">
	<div>
		<TrainControls />
		<EpochReport />
		<LossChart values={$genLossHistory} name="Generator" />
		<LossChart values={$discLossHistory} name="Discriminator" />
	</div>
</Section>

<style>
	div {
		display: grid;
		gap: 1em;
		padding: 1em;
	}
</style>
