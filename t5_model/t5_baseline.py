import tensorflow as tf
import time
import numpy as np
from util import profile_model, get_model_and_tokenizer


def profile_model_layers(model, encoded_inputs):
    layer_times = []
    for encoded_input in encoded_inputs:
        input_ids = encoded_input['input_ids']
        attention_mask = tf.cast(tf.ones_like(input_ids), dtype=tf.float32)
        decoder_input_ids = encoded_input['input_ids']
        decoder_attention_mask = tf.cast(tf.ones_like(decoder_input_ids), dtype=tf.float32)

        input_layer_times = []
        hidden_states = model.shared(input_ids)

        for i, layer in enumerate(model.encoder.block):
            start_time = time.time()
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
            )
            end_time = time.time()
            input_layer_times.append(end_time - start_time)
            hidden_states = layer_output[0]

        layer_times.append(input_layer_times)
    return layer_times


def t5_baseline(test_dataset):
    model, tokenizer = get_model_and_tokenizer()

    # Encoding the inputs
    encoded_inputs = [tokenizer(text, return_tensors='tf') for text in test_dataset]

    # Baseline duration
    baseline_durations = profile_model(model, encoded_inputs)
    average_baseline_duration = np.mean(baseline_durations)

    # Profile each layer and get average layer durations
    layer_durations = profile_model_layers(model, encoded_inputs)
    average_layer_durations = np.mean(layer_durations, axis=0)

    # Assuming 1 unit of power consumption
    power_consumption = 1.0
    average_edp_values = [duration * power_consumption for duration in average_layer_durations]

    return average_baseline_duration, average_layer_durations, average_edp_values
