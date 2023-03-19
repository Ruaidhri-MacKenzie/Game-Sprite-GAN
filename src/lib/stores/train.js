import { writable } from "svelte/store";

export const worker = writable(null);
export const training = writable(false);
export const step = writable(0);
export const genLossHistory = writable([]); // let genLossHistory = Array(100).fill(0).map(y => Math.random() * 100 + 50).map((y, x) => ({ x, y, }));
export const discLossHistory = writable([]);
