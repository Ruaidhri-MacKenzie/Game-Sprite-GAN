import { writable } from "svelte/store";

export const inputShape = writable([32, 32, 4]);
export const patchShape = writable([16, 16, 1]);

export const generator = writable(null);
export const discriminator = writable(null);
export const gan = writable(null);

export const genLearningRate = writable(0.00001);
export const discLearningRate = writable(0.0001);
