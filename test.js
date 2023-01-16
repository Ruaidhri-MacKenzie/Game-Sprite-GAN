import * as tf from "@tensorflow/tfjs-node";
import path from "path";
import mnist from "mnist";

const BUFFER_SIZE = 60000;
const BATCH_SIZE = 256;
const EPOCHS = 50;
const noiseDim = 100;
const numExamplesToGenerate = 16;
const seed = tf.randomNormal([numExamplesToGenerate, noiseDim]);

const checkpointDir = "./training_checkpoints";
const checkpointPrefix = path.join(checkpointDir, "ckpt");

const lr = 0.0001;
const generatorOptimizer = tf.train.adam({ lr });
const discriminatorOptimizer = tf.train.adam({ lr });

const getCurrentTime = () => {
	return (new Date()).getTime() / 1000;
};

const zip = (array1, array2) => {
	const zipped = [];
	for (let i = 0; i < array1.length; i++) zipped.push([array1[i], array2[i]]);
	return zipped;
};

const compareArrays = (array1, array2) => {
	return array1.every((value, index) => value === array2[index]);
};

const assertOutputShape = (model, shape) => {
	const sameShape = compareArrays(model.outputShape, shape);
	tf.util.assert(sameShape, "Model has unexpected output shape.")
};

const makeGeneratorModel = () => {
	const model = tf.sequential();

	model.add(tf.layers.dense({ units: 7 * 7 * 256, useBias: false, inputShape: [100] }));
	model.add(tf.layers.batchNormalization());
	model.add(tf.layers.leakyReLU());
	
	model.add(tf.layers.reshape({ targetShape: [7, 7, 256] }));
	assertOutputShape(model, [null, 7, 7, 256]);
	
	model.add(tf.layers.conv2dTranspose({ filters: 128, kernelSize: [5, 5], strides: [1, 1], padding: "same", useBias: false }));
	assertOutputShape(model, [null, 7, 7, 128]);
	model.add(tf.layers.batchNormalization());
	model.add(tf.layers.leakyReLU());
	
	model.add(tf.layers.conv2dTranspose({ filters: 64, kernelSize: [5, 5], strides: [2, 2], padding: "same", useBias: false }));
	assertOutputShape(model, [null, 14, 14, 64]);
	model.add(tf.layers.batchNormalization());
	model.add(tf.layers.leakyReLU());
	
	model.add(tf.layers.conv2dTranspose({ filters: 1, kernelSize: [5, 5], strides: [2, 2], padding: "same", useBias: false, activation: "tanh" }));
	assertOutputShape(model, [null, 28, 28, 1]);
	
	return model;
};

const makeDiscriminatorModel = () => {
	const model = tf.sequential();

	model.add(tf.layers.conv2d({ filters: 64, kernelSize: [5, 5], strides: [2, 2], padding: "same", inputShape: [28, 28, 1] }));
	model.add(tf.layers.leakyReLU());
	model.add(tf.layers.dropout(0.3));
	model.add(tf.layers.conv2d({ filters: 128, kernelSize: [5, 5], strides: [2, 2], padding: "same" }));
	model.add(tf.layers.leakyReLU());
	model.add(tf.layers.dropout(0.3));
	model.add(tf.layers.flatten());
	model.add(tf.layers.dense({ units: 1 }));

	return model;
};

const discriminatorLoss = (realOutput, fakeOutput) => {
	const realLoss = tf.metrics.binaryCrossentropy(tf.onesLike(realOutput), realOutput);
	const fakeLoss = tf.metrics.binaryCrossentropy(tf.zerosLike(fakeOutput), fakeOutput);
	const totalLoss = realLoss + fakeLoss;
	return totalLoss;
};

const generatorLoss = (fakeOutput) => {
	const genLoss = tf.metrics.binaryCrossentropy(tf.onesLike(fakeOutput), fakeOutput);
	return genLoss;
};

const generateAndSaveImages = (model, epoch, testInput) => {
  // training is set to false. This is so all layers run in inference mode (batchnorm).

	const predictions = model.predict(testInput);
	console.log(predictions);
	// freeze training?
	// const predictions = model.predict(testInput, training=false);
	// unfreeze training?

	// const fig = plt.figure(figsize=(4, 4));

	// for (let i = 0; i < predictions.shape[0]; i++) {
	// 	plt.subplot(4, 4, i + 1);
	// 	plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray");
	// 	plt.axis("off");
	// }

  // plt.savefig(`image_at_epoch_${epoch}.png`);
  // plt.show();
};

const trainStep = async (generator, discriminator, images) => {
	const noise = tf.randomNormal([BATCH_SIZE, noiseDim]);
	
	const generatedImages = await generator.predict(noise);
	
	const realOutput = await discriminator.predict(tf.tensor4d(images, [1, 28, 28, 1]));
	const fakeOutput = await discriminator.predict(generatedImages);
	
	const genLoss = generatorLoss(fakeOutput);
	const discLoss = discriminatorLoss(realOutput, fakeOutput);
	
	// const generatorGradients = generatorOptimizer.computeGradients(genLoss, generator.trainableVariables);
	// const discriminatorGradients = discriminatorOptimizer.computeGradients(discLoss, discriminator.trainableVariables);
	
	generatorOptimizer.applyGradients(zip(genLoss, generator.trainableVariables));
	discriminatorOptimizer.applyGradients(zip(discLoss, discriminator.trainableVariables));
};

const train = async (generator, discriminator, dataset, epochs, checkpoint) => {
	console.log("Training...");
	for (let epoch = 0; epoch < epochs; epoch++) {
		console.log(`Epoch: ${epoch + 1}`);

		const start = getCurrentTime();
		
    for (let imageBatch of dataset) {
			await trainStep(generator, discriminator, imageBatch);
		}
		
    // Produce images for the GIF while training
    // display.clear_output(wait=True)
    generateAndSaveImages(generator, epoch + 1, seed);
		
    // Save the model every 15 epochs
    if ((epoch + 1) % 15 === 0) {
			// checkpoint.save(file_prefix = checkpointPrefix);
		}
		
    console.log(`Time for epoch ${epoch + 1} is ${(getCurrentTime() - start).toFixed(2)} sec`);
	}

  // Generate after the final epoch
  // display.clear_output(wait=True)
  generateAndSaveImages(generator, epochs, seed);
};

// Display a single image using the epoch number
const displayImage = (epoch) => {
  return PIL.Image.open(`image_at_epoch_${epoch}.png`);
};

const main = async () => {
	// Check Tensorflow is using the GPU
	console.log(`Backend: ${tf.getBackend()}`);
	
	/* IMPORT DATA */

	// Load MNIST dataset and split into trainX, trainY, testX, testY
	const set = mnist.set(8000, 2000);
	const trainingSet = set.training;
	const testSet = set.test;
	const trainX = trainingSet.map(instance => instance.input);
	const trainY = trainingSet.map(instance => instance.output);
	const testX = testSet.map(instance => instance.input);
	const testY = testSet.map(instance => instance.output);

	// // Reshape images to match input layer shape
	// trainImages = trainImages.reshape([trainImages.shape[0], 28, 28, 1]).astype("float32");
	
	// // Normalise image values from between 0 and 255 to between -1 and 1
	// // trainImages = trainImages.map(image => (image - 127.5) / 127.5);
	// trainImages = (trainImages - 127.5) / 127.5;
	
	// // Batch and shuffle the dataset
	// const trainDataset = tf.data.Dataset.from_tensor_slices(trainImages).shuffle(BUFFER_SIZE).batch(BATCH_SIZE);
		const trainDataset = trainX;

	/* CREATE MODEL */
	
	// Create generator model
	const generator = makeGeneratorModel();
	
	// Generate random noise
	const noise = tf.randomNormal([1, 100]);
	
	// Generate image using noise
	const generatedImage = generator.predict(noise);
	// freeze training?
	// const generatedImage = generator(noise, training=false);
	// unfreeze training?
	
	// Plot the generated image
	// plt.imshow(generatedImage[0, :, :, 0], cmap="gray");
	
	// Create discriminator model
	const discriminator = makeDiscriminatorModel();
	
	// Generate prediction from discriminator on generated image
	const decision = discriminator.predict(generatedImage);
	// console.log(decision);
	
	// Create initial checkpoint
	// const checkpoint = tf.train.Checkpoint(generatorOptimizer, discriminatorOptimizer, generator, discriminator);
	const checkpoint = null;
	
	/* TRAIN THE MODEL */
	
	await train(generator, discriminator, trainDataset, EPOCHS, checkpoint);
	// checkpoint.restore(tf.train.latest_checkpoint(checkpointDir));
	//displayImage(EPOCHS);
	console.log("Done");

	// // Display animation of generation attempts across epochs
	// const animFile = "dcgan.gif";
	// with imageio.get_writer(animFile, mode='I') as writer:
	//   filenames = glob.glob('image*.png')
	//   filenames = sorted(filenames)
	//   for filename in filenames:
	//     image = imageio.imread(filename)
	//     writer.append_data(image)
	//   image = imageio.imread(filename)
	//   writer.append_data(image)
	
	// embed.embed_file(anim_file)
};

main();
