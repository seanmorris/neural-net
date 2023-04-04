const Neuron = require('./Neuron.js');

module.exports = class Layer
{
	constructor({size, previous, network, offset, idOffset})
	{
		this.idOffset = idOffset;
		this.network  = network;
		this.offset   = offset;

		this.size = size;

		if(previous)
		{
			this.previous = previous;
			previous.next = this;
		}
	}

	populate()
	{
		this.neurons = Array(this.size).fill(null).map((_,i) => new Neuron({
			idOffset: this.idOffset,
			network: this.network,
			offset: this.offset,
			layer: this,
		}));

		if(this.previous)
		{
			this.neurons.forEach(neuron => this.previous.neurons.forEach(
				other => other.connect(neuron)
			));
		}
	}
}