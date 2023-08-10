import * as tf from "@tensorflow/tfjs";

export const getBatch = (dataset, inputShape, index = 0, batchSize = 1) => {
	// Returns 4d tensor
	return tf.tidy(() => {
		return dataset.slice([index, ...Array(inputShape.length).fill(0)], [batchSize, ...inputShape]);
	});
};

export const getSample = (dataset, inputShape, index = 0) => {
	// Returns 3d tensor
	return tf.tidy(() => {
		return getBatch(dataset, inputShape, index, 1);
	});
};

export const imageToSprite = async (url, imageChannels) => {
	const image = new Image();
	image.src = url;
	await new Promise((resolve, reject) => {
		image.onload = () => resolve(image);
		image.onerror = reject;
	});

	return tf.tidy(() => tf.browser.fromPixels(image, imageChannels));
};

export const spriteToInput = (sprite, inputShape) => {
	return tf.tidy(() => {
		// Pad to model input shape
		sprite = sprite.pad([[0, inputShape[0] - sprite.shape[0]], [0, inputShape[1] - sprite.shape[1]], [0, inputShape[2] - sprite.shape[2]]], 0);
		
		// Normalise values to [-1,1]
		sprite = sprite.div(255 / 2).sub(1);

		// Add batch dimension
		sprite = tf.expandDims(sprite);

		return sprite;
	});
};

export const outputToSprite = (output, spriteShape) => {
	return tf.tidy(() => {
		// Crop to sprite shape
		output = output.slice([0, 0, 0, 0], [1, ...spriteShape]);
		
		// Remove batch dimension
		output = output.reshape(spriteShape);
		
		// Normalise values to [0,1]
		output = output.add(1).div(2);

		return output;
	});
};

export const spriteToImage = async (sprite) => {
	const canvas = document.createElement("canvas");
	canvas.width = sprite.shape[1];
	canvas.height = sprite.shape[0];
	await tf.browser.toPixels(sprite, canvas);
	sprite.dispose();
	return canvas.toDataURL();
};

export const normalisePixelValues = (image) => {
	return tf.tidy(() => image.div(255 / 2).sub(1));
};

export const splitSpritesheet = (sprites, spriteShape) => {
	return tf.tidy(() => {
		const [height, width, channels] = spriteShape;
		const count = sprites.shape[1] / width;
		sprites = sprites.reshape([height, count, width, channels]);
		sprites = sprites.transpose([1, 0, 2, 3]);
		sprites = sprites.reshape([count, height, width, channels]);
		return sprites;
	});
};

export const getMinValue = (tensor) => {
	return tf.tidy(() => tf.min(tensor).dataSync());
};

export const getMaxValue = (tensor) => {
	return tf.tidy(() => tf.max(tensor).dataSync());
};

export const padInput = (sprite, padShape) => {
	return tf.tidy(() => sprite.pad([[0, 0], [0, padShape[0]], [0, padShape[1]], [0, padShape[2]]], 0));
};

export const cropInput = (sprite, cropShape) => {
	return tf.tidy(() => sprite.slice([0, 0, 0, 0], [-1, ...cropShape]));
};

export const applyBatchJitter = (source, target) => {
	return tf.tidy(() => {
		const batchSize = source.shape[0];
		const jitter = 0.1;

		// Apply random rotation within the jitter range
		const angles = tf.randomUniform([batchSize], -jitter, jitter);
		const rotatedSource = tf.image.rotateWithOffset(source, angles);
		const rotatedTarget = tf.image.rotateWithOffset(target, angles);
		const newSource = source.concat(rotatedSource);
		const newTarget = target.concat(rotatedTarget);
		return [newSource, newTarget];
	});
};
