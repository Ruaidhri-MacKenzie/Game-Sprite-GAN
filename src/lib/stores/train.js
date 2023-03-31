import { writable } from "svelte/store";

export const training = writable(false);
export const epochs = writable(100);
export const genLossHistory = writable([]);
export const discLossHistory = writable([]);
