<script>
	export let tests;
</script>

<ul>
	{#each tests as epochTests}
		<li>
			{#each epochTests as epochTestImage}
				<img src={epochTestImage} alt="Test preview" />
			{/each}
		</li>
	{/each}
</ul>

<style>
	ul {
		list-style-type: none;
	}
</style>
