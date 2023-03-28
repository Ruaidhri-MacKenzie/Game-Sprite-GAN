export const getSample = (dataset, index = 0) => {
	return tf.tidy(() => {
		return dataset.slice([index, 0, 0, 0], [1, -1, -1, -1]);
	});
};

export const renderPreview = (element, dataset, index = 0) => {
	tf.tidy(() => {
		const image = getSample(dataset, index);
		tf.browser.toPixels(image, element);
	});
};
