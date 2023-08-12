import { writable } from "svelte/store";

export const spriteWidth = writable(32);
export const spriteHeight = writable(32);
export const spriteChannels = writable(4);

export const trainData = writable(null);
export const testData = writable(null);

export const testInputs = writable(null);
