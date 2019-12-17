const fetch = require("file-fetch");
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

async function model() {
    let x_train = await fetch('all_input_notes.json').then(res => res.json());
    let y_train = await fetch('all_output_notes.json').then(res => res.json());
    let x = [];
    for (let i in x_train) {
        x.push(x_train[i])
    }
    let y = [];
    for (let i in y_train) {
        y.push(y_train[i])
    }

    const model = tf.sequential({
        layers: [
            // tf.layers.flatten({units: 1, inputShape: [256]}),
            tf.layers.dense({inputShape: [256], units: 100, activation: 'sigmoid'}),
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
        validationSplit: .8,
        callbacks: {onBatchEnd}
    }).then(info => {
        console.log('Final accuracy', info.history.acc);
    });

    return model;
}

function onBatchEnd(batch, logs) {
    console.log('Accuracy', logs.acc);
}

model()
    .then(model => model.save('file://./model-1c'))
    .then(res => console.error('res:', res));
