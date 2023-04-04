const testingData  = require('./mnist_handwritten_test.json');
const trainingData = require('./mnist_handwritten_train.json');
// const storedModel  = require('./digits.json');

const Neuron  = require('./Neuron.js');
const Layer   = require('./Layer.js');
const Network = require('./Network.js');

const fs = require('node:fs');

const network = new Network({sizes:[784, 512, 256, 128, 10]});

const shuffle = array => {
    for (let i = array.length - 1; i > 0; i--)
	{
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}
const transform = item => ({
	inputs: item.image.map(p => p / 255)
	, outputs: Array(10).fill(0).map((_,i) => Number(i === item.label))
});

let checked = 0, found = 0;

const report = (network, input) => {
	// for(const i in input.image)
	// {
	// 	const p = input.image[i];

	// 	process.stdout.write('\033' + `[48;2;${p};${p};${p}m` + '  \033[0m');

	// 	if(i % 28 === 27)
	// 	{
	// 		process.stdout.write("\n");
	// 	}
	// }

	const transformed = transform(input);
	const output = network.activate(transformed.inputs);

	const digits = [...Object.entries(output)]
	.map(([digit,confidence]) => ({digit, confidence}))
	.sort((a, b) => a.confidence > b.confidence ? -1 : 1);

	const top = digits.slice(0, 1).map(d => Number(d.digit));

	checked++;

	if(top.includes(input.label))
	{
		found++;
	}

	const readable = input.label + ' => ' + top.join(', ') + ' ' + Number(found/checked*100).toFixed(3) + (top.includes(input.label) ? ' - found! ' : ' ');

	console.log(readable);

	return found/checked;
}

const binFile   = __dirname + '/network.bin';

if(fs.existsSync(binFile))
{
	network.loadBuffer();
}
else
{
	network.saveBuffer();
}

//*/
let dataset = trainingData.slice();

shuffle(dataset);

dataset = trainingData.slice(0,);

network.train({dataset, poolSize: 8, iterations: 1, rate: 0.005}).then(() => {
	let score = 0;

	let dataset = trainingData;

	shuffle(dataset);

	dataset = trainingData.slice(0,);

	for(const datum of dataset)
	{
		score = report(network, datum);
	}

	network.updateScore(score);

	if(!network.saveBuffer())
	{
		console.error('Reloading model.');
		network.loadBuffer();
	}
});
/*/
network.loadBuffer();

let dataset = trainingData;

shuffle(dataset);

for(const datum of dataset.slice(0, Math.sqrt(dataset.length)))
{
	report(network, datum);
}
//*/