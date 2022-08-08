const tf = require('@tensorflow/tfjs');
const tfn = require("@tensorflow/tfjs-node");
const repl = require('repl')
let { Tokenizer } = require("tokenizers/bindings/tokenizer");
let { promisify } = require('util');

// Load tokenizer
let tokenizer = Tokenizer.fromPretrained("finiteautomata/ner-leg");

tokenizer.setPadding({ maxLength: 512 });

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


const tokenize = async (text) => {
    let _encode = promisify(tokenizer.encode.bind(tokenizer));
    let output = await _encode(text);

    return output
}

const predict = (model, tokenizedInput) => {
    let inputIds = tf.tensor(tokenizedInput.getIds(), undefined, "int32");
    let attentionMask = tf.tensor(tokenizedInput.getAttentionMask(), undefined, "int32");

    let modelInput = {
        "input_ids": inputIds.reshape([1, -1]),
        "attention_mask": attentionMask.reshape([1, -1]),
    }

    return model.predict(modelInput).squeeze(0);
}

let id2label = [
    "O",
    "B-marker",
    "I-marker",
    "B-reference",
    "I-reference",
    "B-term",
    "I-term"
];

const decode = (prediction, tokenizedInput) => {
    // Decode the prediction
    // First, get the prediction for each token

    let tokenPreds = prediction.argMax(1).arraySync();
    let wordIds = tokenizedInput.getWordIds();
    let currentWordId = null;

    console.log(tokenPreds);
    for (let i = 1; i < prediction.shape[0]; ++i) {
        let token = tokenizedInput.getTokens()[i];
        let pred = tokenPreds[i];
        let wordId = wordIds[i];

        if (wordId !== currentWordId) {
            // Starts new word
            currentWordId = wordId;
            console.log(`${token} (${id2label[pred]})`);
        }

    }

}

const testString = 'a. A Person is deemed to be a holder of Registrable Securities whenever such Person owns or is deemed to own of record such Registrable Securities. If the Company receives conflicting instructions, notices or elections from two or more Persons with respect to the same Registrable Securities, the Company shall act upon the basis of instructions, notice or election received from the registered owner of such Registrable Securities.';

const runNERTest = async () => {
    let model = await loadModel();
    console.log("Model loaded");
    console.log("Input shape");
    console.log(model.signature.inputs["input_ids:0"].tensorShape)
    console.log("Output shape");
    console.log(model.signature.outputs.logits.tensorShape);
    let tokenizedInput = await tokenize(testString);

    console.log(tokenizedInput.getTokens());
    console.log(tokenizedInput.getIds());
    console.log(tokenizedInput.getAttentionMask());


    let prediction =  predict(model, tokenizedInput);

    console.log(prediction);
    console.log(Object.keys(prediction));
    console.log(JSON.stringify(prediction));

    decode(prediction, tokenizedInput);

    let pepe = prediction.arraySync();
    const r = repl.start()
    r.context.prediction = prediction;
    r.context.tokenizedInput = tokenizedInput;
    r.context.model = model;
    r.context.pepe = pepe;
}

runNERTest()
