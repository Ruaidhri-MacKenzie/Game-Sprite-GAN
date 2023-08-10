<script>
	import "$lib/styles/global.css";

	let title = "spriteGAN.js";
</script>

<svelte:head>
	<title>{title}</title>
</svelte:head>

<div>
	<header>
		<h1>{title}</h1>
	</header>
	
	<slot />
	
	<footer>
		<p>Ruaidhri MacKenzie &copy; 2023</p>
	</footer>
</div>

<style>
	div {
		min-height: 100vh;
		display: grid;
		grid-template-rows: auto 1fr auto;
	}

	header {
		text-align: center;
		padding: 1em;
	}

	footer {
		text-align: right;
		margin-top: 1em;
		padding: 0.5em;
		background-color: hsl(0 0% 25%);
		color: white;
	}
</style>
