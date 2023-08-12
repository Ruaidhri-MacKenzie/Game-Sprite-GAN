import { writable } from "svelte/store";

export const training = writable(false);
export const step = writable(0);
export const stepsPerEpoch = writable(0);

export const testSources = writable([]);
export const testTargets = writable([]);

const createEpochTests = () => {
	const { subscribe, set, update } = writable([]);
	return {
		subscribe,
		set,
		addTest: (image) => update(current => [image, ...current]),
		reset: () => set([]),
	};
};
export const epochTests = createEpochTests();

const createLog = () => {
	const { subscribe, set, update } = writable([]);
	return {
		subscribe,
		addEntry: (entry) => update(current => [...current, entry]),
		reset: () => set([]),
	};
};
export const epochLog = createLog();

const createLossHistory = () => {
	const { subscribe, set, update } = writable([]);
	return {
		subscribe,
		addLoss: (epoch, loss) => update(current => [...current, { x: epoch, y: loss }]),
		reset: () => set([]),
	};
};
export const genLossHistory = createLossHistory();
export const discLossHistory = createLossHistory();
