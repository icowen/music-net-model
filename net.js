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
            tf.layers.dense({inputShape: [256], units: 100, activation: 'sigmoid'}),
            tf.layers.dense({units: 128, activation: 'sigmoid'})
        ]
    });
    model.compile({
        optimizer: 'adam',
        loss: tf.losses.logLoss,
        metrics: ['accuracy']
    });
    model.fit(tf.tensor2d(x), tf.tensor2d(y), {
        epochs: 0,
        batchSize: 1,
        validationSplit: .8,
        callbacks: tf.callbacks.earlyStopping({monitor: 'val_loss'})
    }).then(info => {
        console.log('Final loss:', info.history.val_loss);
    });

    return model;
}

model()
    .then(model => model.save('file://./model-untrained'))
    .then(res => console.error('res:', res));
