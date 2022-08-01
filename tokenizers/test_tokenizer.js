let { Tokenizer } = require("tokenizers/bindings/tokenizer");
let { BPE } = require("tokenizers/bindings/models");
let { promisify } = require('util');
//let tokenizer = new Tokenizer(BPE.init({}, [], { unkToken: "[UNK]" }));

const test = async () => {
    let tokenizer = Tokenizer.fromFile("./tokenizer.json");
    let encode = promisify(tokenizer.encode.bind(tokenizer));
    let output = await encode("Hello, y'all! How are you 😁 ?");
    console.log(output.getTokens());
}

test()