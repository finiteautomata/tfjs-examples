const tf = require('@tensorflow/tfjs');
const tfn = require("@tensorflow/tfjs-node");
let { Tokenizer } = require("tokenizers/bindings/tokenizer");
let { promisify } = require('util');

// Load tokenizer
let tokenizer = Tokenizer.fromPretrained("finiteautomata/ner-leg");

tokenizer.setPadding({ maxLength: 384 });

const loadModel = async () => {
    try{
        const handler = tfn.io.fileSystem("./ner-leg/model.json");
        const model = await tfn.loadGraphModel(handler);
        return model;
    }
    catch(error){
        console.log("There was an error loading the model!")
        console.log(error);
        throw error;
    }
}


const encode = async (text) => {
    let _encode = promisify(tokenizer.encode.bind(tokenizer));
    let output = await _encode(text);

    return output
}

const runNERTest = async () => {
    let output = await encode("This is a test, let's check out how it works");
    console.log(output.getTokens());
    console.log(output.getIds());
    console.log(output.getAttentionMask());

    let inputIds = tf.tensor(output.getIds(), undefined, "int32");
    let attentionMask = tf.tensor(output.getAttentionMask(), undefined, "int32");

    let inputs = {
        "input_ids": inputIds.reshape([1, -1]),
        "attention_mask": attentionMask.reshape([1, -1]),
    }
    let model = await loadModel();

    console.log("Model loaded!");
    let prediction = model.predict(inputs);

    console.log(prediction);
}

runNERTest()