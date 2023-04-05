const testingData  = require('./mnist_handwritten_test.json');
const trainingData = require('./mnist_handwritten_train.json');

const Neuron  = require('./Neuron.js');
const Layer   = require('./Layer.js');
const Network = require('./Network.js');

const fs = require('node:fs');

const { shuffle } = require('./common.js');

const network = new Network({sizes:[784, 512, 256, 128, 10]});

const binFile = __dirname + '/network.bin';

if(fs.existsSync(binFile))
{
	network.loadBuffer(binFile);
}
else
{
	network.saveBuffer(binFile);
}

let dataset = trainingData.slice();

shuffle(dataset);

dataset = dataset.slice(0,);

network.train({dataset, poolSize: 8, iterations: 2, rate: 0.0005}).then(() => {
	let dataset = testingData.slice();

	shuffle(dataset);

	dataset = dataset.slice(0,);

	return network.score({dataset, poolSize: 8});

}).then(score => {

	console.log(score);

	if(!network.saveBuffer(binFile))
	{
		console.error('Bailing on save, previous model had higher score.');
		console.error('Reloading model...');
		network.loadBuffer(binFile);
	}
});
