const tf = require('@tensorflow/tfjs');
const tfn = require("@tensorflow/tfjs-node");

const test = async () => {
    try{
        const handler = tfn.io.fileSystem("./convert_to_tf/foo_js/model.json");

        //const model = await tf.loadLayersModel("/home/jmperez/projects/tfjs-examples/hf_model/robertuito_js/prueba_layers/model.json");
        const model = await tfn.loadGraphModel(handler);
        console.log("LO CARGAMOS!")

        return model;
    }
    catch(error){
        console.log("ERRORRRRRRRRRRRRRRRR!!!!")
        console.log(error);
    }
}

(async () => {
    test();
})();