import { writable } from "svelte/store";

export const inputShape = writable([32, 32, 4]);

export const generator = writable(null);
export const discriminator = writable(null);
export const gan = writable(null);
