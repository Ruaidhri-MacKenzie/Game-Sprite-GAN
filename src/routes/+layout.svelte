<script>
	import "$lib/styles/reset.css";
</script>

<div>
	<header>
		<h1>Game Sprite GAN</h1>
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
		padding: 1em 0.5em 0.5em;
	}
</style>
