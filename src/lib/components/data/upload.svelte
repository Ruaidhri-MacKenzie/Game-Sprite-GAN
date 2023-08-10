<script>
	import { spriteChannels } from "$lib/stores/data.js";
	import { imageToSprite } from "$lib/utils/data.utils.js";
	import TensorInfo from "./tensor-info.svelte";
	import InputUpload from "$lib/components/common/input-upload.svelte";

	export let spritesheet;

	const onUploadSpritesheet = async (event) => {
		spritesheet = await imageToSprite(URL.createObjectURL(event.target.files[0]), $spriteChannels);
	};
</script>

<form>
	<h3>Upload</h3>
	<InputUpload text="Upload Spritesheet" handleChange={onUploadSpritesheet} accept="image/*"/>
	<TensorInfo tensor={spritesheet} />
</form>

<style>
	form {
		display: grid;
		align-items: center;
		gap: 0.5em;
	}

	h3 {
		text-align: center;
	}
</style>
