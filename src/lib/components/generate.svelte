<script>
	import * as tf from "@tensorflow/tfjs";
	import { imageWidth, imageHeight, imageChannels, spriteWidth, spriteHeight, spriteChannels } from "$lib/stores/data.js";
	import { generator } from "$lib/stores/model.js";
	import { downloadModel } from "$lib/actions/model.js";
	
	let sourceImage;
	let generatedImage;

	const spritePipeline = (sprite) => {
		sprite = sprite.reshape([$spriteHeight, $spriteWidth, $spriteChannels]);
		sprite = sprite.div(255 / 2).sub(1);
		sprite = sprite.pad([[0, $imageHeight - $spriteHeight], [0, $imageWidth - $spriteWidth], [0, 0]], 0);
		sprite = sprite.pad([[0, 0], [0, 0],  [0, $imageChannels - $spriteChannels]], 1);
		// sprite.slice([0, 0, 0], [$imageHeight, $imageWidth, $imageChannels]);
		return sprite;
	};

	const uploadSourceAndGenerate = async (event) => {
		const image = new Image();
		image.src = URL.createObjectURL(event.target.files[0]);
    await new Promise((resolve, reject) => {
			image.onload = () => resolve(image);
			image.onerror = reject;
    });
		let source = tf.browser.fromPixels(image, $spriteChannels);
		source = spritePipeline(source);
		source = source.reshape([1, $imageHeight, $imageWidth, $imageChannels]);
		let target = await $generator.predict(source);
		source = source.slice([0, 0, 0, 0], [1, $spriteHeight, $spriteWidth, $spriteChannels]).squeeze().add(1).div(2);
		target = target.slice([0, 0, 0, 0], [1, $spriteHeight, $spriteWidth, $spriteChannels]).squeeze().add(1).div(2);
		tf.browser.toPixels(source, sourceImage);
		tf.browser.toPixels(target, generatedImage);
	};

	const saveGenerator = (event) => {
		downloadModel($generator, "sprite-gan");
	};
</script>

<section>
	<h2>Generate</h2>
	<input on:change={uploadSourceAndGenerate} type="file" name="sourceImage">
	<span>
		<canvas bind:this={sourceImage}></canvas>
		<canvas bind:this={generatedImage}></canvas>
	</span>
	<button on:click={saveGenerator}>Save Generator</button>
</section>

<style>
	section {
		display: grid;
		gap: 1em;
		padding: 1em;
		box-shadow: 0 3px 8px hsl(0 0% 0% / 0.24);
	}

	span {
		display: flex;
		gap: 1em;
		padding: 1em;
	}

	canvas {
		background-color: black;
		width: 64px;
		height: 64px;
	}

	button {
		width: fit-content;
		padding: 0.5em 1em;
	}
</style>
