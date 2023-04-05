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

network.loadBuffer(binFile);

let dataset = trainingData;

shuffle(dataset);

dataset = dataset.slice(0,1000);

network.score(dataset).then(console.log);
