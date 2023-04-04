const { parentPort } = require('node:worker_threads');

const convertBuffer = new ArrayBuffer(4);
const convertView = new DataView(convertBuffer, 0, 4);

const sigmoid  = x => 1 / (1 + Math.exp(-x));
const asigmoid = x => sigmoid(x) * (1 - sigmoid(x));

const int2float = intVal => {
	convertView.setInt32(0, intVal);
	return convertView.getFloat32(0);
};

const float2int = floatVal => {
	convertView.setFloat32(0, floatVal);
	return convertView.getInt32(0);
};

const weightPtr = (neuron, other) => {
	const wStart     = neuron.network.totalSize;
	const idInLayer  = neuron.id + neuron.idOffset;
	const oidInLayer = other.id + other.idOffset;
	const nextSize   = neuron.layer.next
		? neuron.layer.next.size
		: 0;

	return (nextSize * idInLayer) + oidInLayer + neuron.layerOffset + wStart;
};

module.exports = class Neuron
{
	constructor({bias, network, layer, offset, idOffset} = {})
	{
		this.layerOffset = offset;
		this.idOffset = idOffset;
		this.id       = network.lastId++;

		this.incoming = new Set;
		this.outgoing = new Set;

		this.locals = new Float32Array(3); // _output, output, error

		this.initialized = false;
		this.network     = network;
		this.layer       = layer;
		this.bias        = bias;

		this.cells = network.cells;
		this.locks = network.locks;

		if(!this.bias)
		{
			Atomics.store(this.cells, this.id, float2int(this.bias ?? Math.random() * 2 - 1));
		}

		this.bias = int2float(Atomics.load(this.cells, this.id));

		this.sumIncoming = (total, other) => {
			return total + other.locals[1] * int2float(Atomics.load(this.cells, weightPtr(other, this)));
		};
	}

	connect(other, weight)
	{
		this.outgoing.add(other);
		other.incoming.add(this);

		const ptr = weightPtr(this, other);

		const existing = int2float(Atomics.load(this.cells, ptr));

		if(!existing)
		{
			weight = weight ?? Math.random() * 2 - 1;
			Atomics.store(this.cells, ptr, float2int(weight));
		}
	}

	activate(input)
	{
		const locals = this.locals;

		if(input !== undefined)
		{
			locals[0] = 1;
			locals[1]  = input;
		}
		else
		{
			const sum = [...this.incoming].reduce(this.sumIncoming, int2float(Atomics.load(this.cells, this.id)));

			locals[0] = asigmoid(sum);
			locals[1]  = sigmoid(sum);
		}

		return locals[1];
	}

	propagate(target, rate = 0.1)
	{
		this.lockCell(this.id);

		const locals = this.locals;

		const sum = target === undefined
			? [...this.outgoing].reduce((total, other) => {

					const ptr = weightPtr(this, other);

					const otherError = other.locals[2];

					let weight = int2float(Atomics.load(this.cells, ptr));

					weight -= rate * otherError * this.locals[1];

					Atomics.store(this.cells, ptr, float2int(weight));

					return total + otherError * weight;
				},
				0
			)
			: locals[1] - target;

		let bias = int2float(Atomics.load(this.cells, this.id));

		bias -= rate * locals[2];

		Atomics.store(this.cells, this.id, float2int(bias));

		this.unlockCell(this.id);

		locals[2] = sum * locals[0]

		return locals[2];
	}

	waitForCell(ptr)
	{
		Atomics.wait(this.locks, ptr, 1);
	}

	lockCell(ptr)
	{
		this.waitForCell(ptr);
		Atomics.store(this.locks, ptr, 1);
	}

	unlockCell(ptr)
	{
		Atomics.store(this.locks, ptr, 0);
		Atomics.notify(this.locks, ptr, 4);
	}
}