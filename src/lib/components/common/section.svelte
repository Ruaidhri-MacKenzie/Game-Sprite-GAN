<script>
	export let title;
</script>

<section>
	<h2>{title}</h2>
	<slot />
</section>

<style>
	section {
		box-shadow: 0 3px 8px hsl(0 0% 0% / 0.24);
	}
	
	h2 {
		text-align: center;
		padding: 0.25em 0.5em;
		background-color: var(--color-primary);
		color: var(--color-contrast);
		user-select: none;
		cursor: default;
	}
</style>
