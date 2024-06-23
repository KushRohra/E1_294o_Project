import tensorflow as tf
import time
from util import get_model_and_tokenizer, profile_model

import tensorflow as tf
import time
import numpy as np
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer

def profile_model_layers(model, encoded_inputs):
    transformer_layers = model.roberta.encoder.layer
    layer_times = []

    for encoded_input in encoded_inputs:
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input.get('attention_mask', tf.ones_like(input_ids))
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)  # Ensure attention_mask is float32
        
        hidden_states = model.roberta.embeddings(input_ids)

        input_layer_times = []
        for layer in transformer_layers:
            start_time = time.time()
            layer_output = layer(
                hidden_states, 
                attention_mask=attention_mask,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=False,
                training=False
            )
            end_time = time.time()
            input_layer_times.append(end_time - start_time)
            hidden_states = layer_output[0]
        layer_times.append(input_layer_times)
    
    return layer_times

def roberta_improved_edp(test_dataset):
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model, tokenizer = get_model_and_tokenizer()

        # Encoding the inputs
        encoded_inputs = [tokenizer(text, return_tensors='tf') for text in test_dataset]

        # Profile each layer and get average layer durations
        layer_durations = profile_model_layers(model, encoded_inputs)
        average_layer_durations = np.mean(layer_durations, axis=0)

        average_layer_durations = np.array(average_layer_durations)

        # Assuming 1 unit of power consumption
        power_consumption = 1.0  
        average_edp_values = [duration * power_consumption for duration in average_layer_durations]

    return average_layer_durations, average_edp_values
