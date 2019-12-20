const fetch = require("file-fetch");
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

async function model() {
    let x_train = await fetch('all_input_notes.json').then(res => res.json());
    let y_train = await fetch('all_output_notes.json').then(res => res.json());
    let numOfTrainingSongs = 10;
    let x = [];
    for (let i = 0; i < numOfTrainingSongs; i++) {
        x.push(x_train[i])
    }
    let y = [];
    for (let i = 0; i < numOfTrainingSongs; i++) {
        y.push(y_train[i])
    }

    const model = tf.sequential({
        layers: [
            // tf.layers.flatten({units: 1, inputShape: [256]}),
            tf.layers.dense({inputShape: [256], units: 256, activation: 'sigmoid'}),
            tf.layers.dense({units: 128, activation: 'sigmoid'})
        ]
    });
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    model.fit(tf.tensor2d(x), tf.tensor2d(y), {
        epochs: 10000,
        batchSize: 1,
        validationSplit: .8
    }).then(info => {
        console.log('Final accuracy', info.history.acc);
    });

    return model;
}

model()
    .then(model => model.save('file://./model-1c'))
    .then(res => console.error('res:', res));
