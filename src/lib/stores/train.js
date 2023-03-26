import { writable } from "svelte/store";

export const training = writable(false);
export const epochs = writable(50);
export const genLossHistory = writable([]); // let genLossHistory = Array(100).fill(0).map(y => Math.random() * 100 + 50).map((y, x) => ({ x, y, }));
export const discLossHistory = writable([]);
