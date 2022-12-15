import { optimizer, loss, metrics, batchSize, epochs } from "./config.js";
import { createGenerator } from "./generator.js";
import { createDiscriminator } from "./discriminator.js";

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

export const trainModel = async (model, xs, ys) => {
	console.log("Training...");
	console.time("Training");
	const results = await model.fit(xs, ys, { batchSize, epochs });
	console.timeEnd("Training");
	return results;
};

export const testModel = async (model, xs, ys) => {
	console.log("Testing...");
	console.time("Testing");
	const results = await model.evaluate(xs, ys);
	console.timeEnd("Testing");
	return results;
};

export const generateImage = async (model, xs) => {
	console.log("Generating...");
	console.time("Generating");
	const results = await model.predict(xs, { batchSize, verbose: true });
	console.timeEnd("Generating");
	return results;
};
