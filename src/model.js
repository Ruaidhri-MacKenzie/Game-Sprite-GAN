import { optimizer, loss, metrics, batchSize, epochs } from "./config.js";
import { createGenerator } from "./generator.js";
import { createDiscriminator } from "./discriminator.js";

export class GAN {
	constructor() {
		this.generator = createGenerator();
		this.discriminator = createDiscriminator();
		this.generatorLosses = [];
		this.discriminatorLosses = [];
	}

	async train(xs, ys) {
		for (let i = 0; i < epochs; i++) {
			// generate fake images from xs
			const fakeImages = await this.generator.predict(xs);
			
			// train discriminator with labelled real/fake images
			const images = tf.concat([ys, fakeImages], 0);
			const labels = tf.concat([tf.ones([ys, 1]), tf.zeros([fakeImages, 1])], 0);
			
			const discriminatorLoss = await discriminator.fit(images, labels, { batchSize, epochs: 1 });
			this.discriminatorLosses.push(discriminatorLoss.loss);
			console.log(`Epoch ${i}: Discriminator loss = ${discriminatorLoss.loss}`);

			// train generator with xs (features/source image) and ys (label/target image)
			const generatorLoss = await generator.fit(xs, ys, { batchSize, epochs: 1 });
			this.discriminatorLosses.push(generatorLoss.loss);
			console.log(`Epoch ${i}: Generator loss = ${generatorLoss.loss}`);

			// apply discriminator gradient to generator
			
		}
	}

	async test(generated, target) {
		// FID
	}

	async generate(xs) {
		const generatedImage = await this.generator.predict(xs);
		return generatedImage;
	}
}

export const createGAN = () => {
	// Create generator and discriminator models
	const generator = createGenerator();
	const discriminator = createDiscriminator();
	
	// Freeze the weights of the discriminator
	discriminator.trainable = false;
	
	// Build the combined model
	const combined = tf.sequential();
	combined.add(generator);
	combined.add(discriminator);
	combined.compile({ optimizer, loss }, metrics);
	return combined;
};

export const trainModel = async (generator, discriminator, xs, ys) => {
	console.log("Training...");
	console.time("Training");
	// const result = await model.fit(xs, ys, { batchSize, epochs });
	
	// Train the GAN model
	for (let i = 0; i < epochs; i++) {
		// Generate a batch of fake images
		const fakeImages = await generator.predict(xs);

		// Concatenate the real and fake images
		const images = tf.concat([ys, fakeImages], 0);

		// Create the labels for the real and fake images
		const labels = tf.concat([tf.ones([batchSize, 1]), tf.zeros([batchSize, 1])], 0);

		// Train the discriminator
		const discriminatorLoss = await discriminator.fit(images, labels, { batchSize, epochs: 1 });
		console.log(`Epoch ${i}: Discriminator loss = ${discriminatorLoss.loss}`);

		// Train the generator
		const generatorLoss = await generator.fit(xs, tf.ones([batchSize, 1]), { batchSize, epochs: 1 });
		console.log(`Epoch ${i}: Generator loss = ${generatorLoss.loss}`);
	}

	console.timeEnd("Training");
	return result;
};

export const testModel = async (model, xs, ys) => {
	console.log("Testing...");
	console.time("Testing");
	const result = await model.evaluate(xs, ys);
	console.timeEnd("Testing");
	return result;
};

export const generateImage = async (model, xs) => {
	console.log("Generating...");
	console.time("Generating");
	const result = await model.predict(xs, { batchSize, verbose: true });
	console.timeEnd("Generating");
	return result;
};
