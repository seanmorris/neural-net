const { Worker, isMainThread, parentPort, workerData } = require('node:worker_threads');

const Layer  = require('./Layer.js');

const fs = require('node:fs');

const transform = item => ({
	inputs: item.image.map(p => p / 255)
	, outputs: Array(10).fill(0).map((_,i) => Number(i === item.label))
});

class Network
{
	constructor({sizes, cells, locks})
	{
		this.sizes   = sizes;
		this.neurons = new Map;
		this.hiddens = new Set;
		this.lastId  = 0;

		this.totalSize = sizes.reduce((a,b) => a+b,0);
		this.lastScore = 0;
		this.score     = 0;

		let lastLayer, biasSize = 0, connSize = 0, last = 0;

		for(const size of sizes)
		{
			const layer = new Layer({
				idOffset: -biasSize,
				previous:lastLayer,
				offset: connSize,
				network:this,
				size,
			});

			if(!this.inputs)
			{
				this.inputs = layer;
			}
			else
			{
				this.hiddens.add(layer);
			}

			connSize += size * last;
			biasSize += size;

			lastLayer = layer;
			last      = size;
		}

		const bufferType = typeof SharedArrayBuffer !== 'undefined'
			? SharedArrayBuffer
			: ArrayBuffer;

		this.cells = cells ?? new Int32Array(new bufferType((biasSize + connSize) * 4));
		this.locks = locks ?? new Int32Array(new bufferType((biasSize + connSize) * 4));

		this.hiddens.delete(lastLayer);
		this.outputs = lastLayer;

		for(const layer of [this.inputs, ...this.hiddens, this.outputs])
		{
			layer.populate();

			layer.neurons.forEach(n => this.neurons.set(n.id, n));
		}
	}

	activate(input)
	{
		this.inputs.neurons.forEach((neuron, i) => neuron.activate(input[i]));

		this.hiddens.forEach(layer => layer.neurons.forEach(neuron => neuron.activate()));

		return this.outputs.neurons.map(neuron => neuron.activate());
	}

	propagate(target, rate = 0.3)
	{
		this.outputs.neurons.forEach((neuron, t) => neuron.propagate(target[t], rate));

		this.hiddens.forEach(layer => layer.neurons.forEach(neuron => neuron.propagate(undefined, rate)));

		return this.inputs.neurons.map(neuron => neuron.propagate(undefined, rate));
	}

	loadBuffer(force = false)
	{
		const binFile = __dirname + '/network.bin';
		const bin     = fs.readFileSync(binFile);

		const aBuffer = new ArrayBuffer(bin.length);
		const uInts   = new Uint8Array(aBuffer);
		const iInts   = new Int32Array(aBuffer);
		const floats  = new Float32Array(aBuffer);

		for(let i = 0; i < bin.length; i++)
		{
			uInts[i] = bin[i];
		}

		const newScore = floats[0];

		if(!force && newScore < this.score)
		{
			console.error('Bailing on load, current model has higher score.');

			return;
		}

		this.score = floats[0];

		this.cells.set(iInts.slice(1));
	}

	updateScore(score)
	{
		this.lastScore = this.score;

		this.score = score;
	}

	saveBuffer(force = false)
	{
		if(!force && this.lastScore > this.score)
		{
			console.error('Bailing on save, previous model had higher score.');

			return false;
		}

		const binFile  = __dirname + '/network.bin';
		const buffer   = Buffer.alloc(4 + this.cells.buffer.byteLength);
		const uInts    = new Uint8Array(this.cells.buffer);
		const floats   = new Float32Array([this.score]);
		const scoreInts = new Uint8Array(floats.buffer);

		for(let i = 0; i < scoreInts.length; i++)
		{
			buffer[i] = scoreInts[i];
		}

		for(let i = 0; i < buffer.length; i++)
		{
			buffer[4 + i] = uInts[i];
		}

		fs.writeFileSync(binFile, buffer);

		return true;
	}

	train({dataset, iterations = 1, rate = 0.3, poolSize = 8})
	{
		const chunkSize = Math.ceil(dataset.length / poolSize);

		const iterate = iterations => {
			if(iterations <= 0)
			{
				return Promise.resolve();
			}

			const promises  = [];

			console.error(`${iterations} training iterations remaining.`);

			for(let i = 0; i < poolSize; i++)
			{
				const promise = new Promise((accept, reject) => {
					const worker = new Worker(__filename, {workerData: {
						sizes: this.sizes,
						cells: this.cells,
						locks: this.locks,
						iterations,
						childId: i,
						poolSize,
						dataset: dataset.slice(i * chunkSize, (1 + i) * chunkSize),
						rate,
					}});

					worker.on('message', console.error);
					worker.on('error', error => reject(error));
					worker.on('exit',  code  => code ? reject(code) : accept());
				});

				promise.catch(console.error);

				promises.push(promise);
			}

			return Promise.all(promises).then(() => iterate(-1 + iterations));
		}

		return iterate(iterations);
	}
}

if(isMainThread)
{
	module.exports = Network;
}
else
{
	const network = new Network(workerData);

	let p    = 0;

	const iterations = workerData.iterations;
	const poolSize   = workerData.poolSize;
	const dataset    = workerData.dataset;

	const start  = Date.now() / 1000;
	const rate   = workerData.rate;
	const id     = workerData.childId;

	const reportMod = 10;

	let i = 0;
	let last = Date.now() / 1000;

	const max = dataset.length;

	while(dataset.length)
	{
		const datum = transform(dataset.pop());

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
}