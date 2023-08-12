<script>
	export let checked = false;
</script>

<span class:checked>{(checked) ? "✓" : "X"}</span>

<style>
	span {
		color: red;
	}

	span.checked {
		color: green;
	}
</style>
