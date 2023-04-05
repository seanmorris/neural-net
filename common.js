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

module.exports = { shuffle, transform };
