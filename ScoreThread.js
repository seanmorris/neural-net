const { parentPort, workerData } = require('node:worker_threads');

const Network = require('./Network.js');

const network  = new Network(workerData);

const poolSize = workerData.poolSize;
const childId  = workerData.childId;
const dataset  = workerData.dataset;

const { transform } = require('./common.js');

let score = 0, checked = 0, found = 0;

for(const datum of dataset)
{
	const transformed = transform(datum);
	const output = network.activate(transformed.inputs);

	const digits = [...Object.entries(output)]
	.map(([digit,confidence]) => ({digit, confidence}))
	.sort((a, b) => a.confidence > b.confidence ? -1 : 1);

	const top = digits.slice(0, 1).map(d => Number(d.digit));

	checked++;

	if(top.includes(datum.label))
	{
		found++;
	}

	score = found/checked;

	if(checked && checked % 100 === 0)
	{
		const readable = `${poolSize}::${childId} | `
			+ `${found} / ${dataset.length}` + ' | '
			+ Number(score*100).toFixed(3) + '% | '
			+ (score > network.ratio ? ' * ' : '   ')
		;

		parentPort.postMessage(readable);
	}
}

parentPort.postMessage({score});
