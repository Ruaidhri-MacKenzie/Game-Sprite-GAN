<script>
	export let sources;
	export let targets;
	export let tests;
</script>

<ul>
	<span>
		<li>
			<p>Source:</p>
			{#each sources as source}
				<img src={source} alt="source test" />
			{/each}
		</li>
		<li>
			<p>Target:</p>
			{#each targets as target}
				<img src={target} alt="target test" />
			{/each}
		</li>
	</span>
	<div>
		{#each tests as epochTests, i}
			<li>
				<p>Epoch {(tests.length - 1) - i}:</p>
				{#each epochTests as epochTestImage}
				<img src={epochTestImage} alt="test" />
				{/each}
			</li>
		{/each}
	</div>
</ul>

<style>
	ul {
		display: flex;
		flex-direction: column;
		gap: 1em;
		list-style-type: none;
	}

	span {
		margin-right: 17px;
		display: flex;
		flex-direction: column;
		gap: 1em;
	}

	div {
		height: 20em;
		overflow-y: scroll;
		display: flex;
		flex-direction: column;
		gap: 1em;
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
	}
</style>
