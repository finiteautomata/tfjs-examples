## Converting huggingface model to TF.js model

0. Install environment

```bash
npm i
cd convert_to_tf && poetry install . && poetry shell
```

1. Save model as `tf_saved_model` format

```python
import tensorflow as tf
from transformers import TFDistilBertForTokenClassification, TFRobertaForSequenceClassification

model_name = "pysentimiento/robertuito-sentiment-analysis"
model = TFRobertaForSequenceClassification.from_pretrained("pysentimiento/robertuito-sentiment-analysis", from_pt=True)
model._set_inputs(tf.TensorSpec([1, 384], tf.int32))
tf.saved_model.save(model, model_name +'_js')
```

Check [this notebook](https://colab.research.google.com/drive/18cW5RCCBGYcYvh9BHfYbsmFoJElosqsc#scrollTo=6OWQR6wVD6EV) for more information.

2. Use tfjs-converter to convert model to `tfjs_graph_model` format

```bash
tensorflowjs_converter  \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model\
    ner-leg ner-leg_js
```

3. Run `load_model.js`


## Readings

["Insights of porting tokenizers to WASM"](https://blog.mithrilsecurity.io/porting-tokenizers-to-wasm/)
["Transformers serialization to ONNX"](https://huggingface.co/docs/transformers/serialization)