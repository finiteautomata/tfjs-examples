const tf = require('@tensorflow/tfjs');
const tfn = require("@tensorflow/tfjs-node");

const test = async () => {
    const handler = tfn.io.fileSystem("./convert_to_tf/prueba_layers/model.json");
    console.log(handler);
    //const model = await tfn.loadLayersModel("/home/jmperez/projects/tfjs-examples/hf_model/convert_to_tf/prueba_layers/model.json");
    const model = await tfn.loadLayersModel(handler);
}

(async () => {
    test();
})();