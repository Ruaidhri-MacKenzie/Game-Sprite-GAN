<script>
	export let sources;
	export let targets;
	export let tests;
</script>

<div>
	<ul class="example">
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
	</ul>

	<ul class="test">
		{#each tests as epochTests, i}
			<li>
				<p>Epoch {(tests.length - 1) - i}:</p>
				{#each epochTests as epochTestImage}
					<img src={epochTestImage} alt="test" />
				{/each}
			</li>
		{/each}
	</ul>
</div>

<style>
	div {
		display: flex;
		flex-direction: column;
		gap: 1em;
		list-style-type: none;
		border: 1px solid black;
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
	}
</style>
