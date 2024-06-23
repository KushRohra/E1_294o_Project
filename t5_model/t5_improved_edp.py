import tensorflow as tf
import time
from util import get_model_and_tokenizer


def profile_model_layers_with_strategy(model, encoded_inputs, strategy):
    num_layers = len(model.encoder.block)
    total_layer_times = [0.0] * num_layers
    num_inputs = len(encoded_inputs)

    for encoded_input in encoded_inputs:
        with strategy.scope():
            input_ids = encoded_input['input_ids']
            attention_mask = tf.cast(tf.ones_like(input_ids), dtype=tf.float32)
            decoder_input_ids = encoded_input['input_ids']
            decoder_attention_mask = tf.cast(tf.ones_like(decoder_input_ids), dtype=tf.float32)

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
                layer_duration = end_time - start_time
                total_layer_times[i] += layer_duration
                hidden_states = layer_output[0]

    average_layer_times = [total_layer_time / num_inputs for total_layer_time in total_layer_times]

    return average_layer_times


def t5_improved_edp(test_dataset):
    model, tokenizer = get_model_and_tokenizer()

    # Encoding the inputs
    encoded_inputs = [tokenizer(text, return_tensors='tf') for text in test_dataset]

    # Set up the distribution strategy
    strategy = tf.distribute.MirroredStrategy()

    # Profile each layer using the distribution strategy
    average_layer_durations = profile_model_layers_with_strategy(model, encoded_inputs, strategy)

    # Assuming 1 unit of power consumption
    power_consumption = 1.0
    average_edp_values = [duration * power_consumption for duration in average_layer_durations]

    return average_layer_durations, average_edp_values
