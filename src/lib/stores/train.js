import { writable } from "svelte/store";

export const training = writable(false);
export const epochs = writable(50);

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
