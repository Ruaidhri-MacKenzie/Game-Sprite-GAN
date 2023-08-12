export const epochReportToString = (report) => {
	return `${report.epoch}/${report.epochs}: ${report.time.toFixed(2)}sec | Generator loss: ${report.generatorLoss.toFixed(4)} - Discriminator loss: ${report.discriminatorLoss.toFixed(4)}`;
};

export const printEpochReport = (report) => {
	console.log(epochReportToString(report));
};
