const { parentPort, workerData } = require('node:worker_threads');

const Network = require('./Network.js');

const network = new Network(workerData);

const { transform } = require('./common.js');

let p = 0;

const iterations = workerData.iterations;
const poolSize   = workerData.poolSize;
const dataset    = workerData.dataset;

const start  = Date.now() / 1000;
const rate   = workerData.rate;
const id     = workerData.childId;

const reportMod = 100;

let i = 0;
let last = Date.now() / 1000;

const max = dataset.length;

while(dataset.length)
{
	const datum = transform(dataset.shift());

	network.activate(datum.inputs);
	network.propagate(datum.outputs, rate);

	if(i && i % reportMod === 0)
	{
		const now   = Date.now() / 1000;
		const diff  = now - last || Number.EPSILON;
		const speed = (reportMod / diff);
		const done  = ((1+i) / max);
		const time  = now - start;
		const left  = (time / done) + -time;

		const minutes = Math.trunc(left / 60);
		const seconds = Math.trunc(left % 60);

		parentPort.postMessage(
			`${-iterations}::${poolSize}::${id}  |  `
			+ `${speed.toFixed(2)} i/s  |  `
			+ `${diff.toFixed(2)}s  |  `
			+ `${(done * 100).toFixed(2)}%  |  `
			+ `${i} / ${max}  |  `
			+ `${minutes}:${String(seconds).padStart(2, 0)}`
		);

		last = now;
	}

	i++;
}
