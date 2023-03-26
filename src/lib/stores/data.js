import { writable } from "svelte/store";

export const imageWidth = writable(32);
export const imageHeight = writable(32);
export const imageChannels = writable(4);

export const spriteWidth = writable(32);
export const spriteHeight = writable(32);
export const spriteChannels = writable(4);

export const dataset = writable(null);
