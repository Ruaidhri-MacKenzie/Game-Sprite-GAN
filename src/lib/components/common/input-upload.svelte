<script>
	export let text = "";
	export let accept = "*";
	export let handleChange = () => {};

	let element;

	const handleClick = (event) => {
		if (element) element.click();
	};
</script>

<button on:click={handleClick}>
	<p>{text}</p>
	<input type="file" bind:this={element} on:change={handleChange} {accept}>
</button>

<style>
	input {
		display: none;
	}
</style>
