<script>
	import { testSources, testTargets, epochTests } from "$lib/stores/train.js";
	import { backgroundColour } from "$lib/stores/style.js";
</script>

<div style="--background: {$backgroundColour}">
	<label>
		Background:
		<input type="color" bind:value={$backgroundColour}>
	</label>

	<ul class="example">
		<li>
			<p>Source:</p>
			{#each $testSources as source}
				<img src={source} alt="source test" />
			{/each}
		</li>

		<li>
			<p>Target:</p>
			{#each $testTargets as target}
				<img src={target} alt="target test" />
			{/each}
		</li>
	</ul>

	{#if $epochTests.length}
		<ul class="test">
			{#each $epochTests as tests, i}
				<li>
					<p>Epoch {($epochTests.length - 1) - i}:</p>
					{#each tests as testImage}
						<img src={testImage} alt="test" />
					{/each}
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	div {
		display: flex;
		flex-direction: column;
		gap: 1em;
		list-style-type: none;
	}

	label {
		padding: 1em;
	}

	input {
		cursor: pointer;
	}

	ul {
		display: flex;
		flex-direction: column;
		gap: 1em;
	}

	ul.example {
		margin-right: 17px;
		padding-block: 1em;
	}

	ul.test {
		height: 20em;
		overflow-y: scroll;
	}

	li {
		display: grid;
		grid-template-columns: 8em repeat(4, 1fr);
		justify-items: center;
		align-items: center;
		gap: 1em;
	}

	p {
		font-size: 1.25rem;
		justify-self: right;
	}

	img {
		width: 64px;
		height: 64px;
		background-color: var(--background);
	}
</style>
