import tensorflow as tf
import time
from util import get_model_and_tokenizer

def profile_model_layers_with_strategy(model, encoded_inputs, strategy):
    num_layers = len(model.transformer.h)
    total_layer_times = [0.0] * num_layers
    num_inputs = len(encoded_inputs)

    for encoded_input in encoded_inputs:
        with strategy.scope():
            # Extract the necessary inputs for the layers
            input_ids = encoded_input['input_ids']
            attention_mask = tf.ones_like(input_ids)

            # Get initial hidden states from the embedding layer
            hidden_states = model.transformer.wte(input_ids) + model.transformer.wpe(tf.range(input_ids.shape[-1]))

            for i, layer in enumerate(model.transformer.h):
                start_time = time.time()

                # Forward pass through the specific layer
                layer_output = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    layer_past=None,
                    use_cache=False,
                    output_attentions=False
                )

                end_time = time.time()

                # Calculate the duration for the layer
                layer_duration = end_time - start_time
                total_layer_times[i] += layer_duration

                # Update the input for the next layer
                hidden_states = layer_output[0]  # last_hidden_state

    # Average the layer times over the number of inputs
    average_layer_times = [total_layer_time / num_inputs for total_layer_time in total_layer_times]

    return average_layer_times

def gpt2_improved_edp(test_dataset):
    # Ensure TensorFlow is using GPU if available
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    model, tokenizer = get_model_and_tokenizer()

    # Encoding the inputs
    encoded_inputs = [tokenizer(text, return_tensors='tf') for text in test_dataset]

    # Set up the distribution strategy
    strategy = tf.distribute.MirroredStrategy()

    # Profile each layer using the distribution strategy
    average_layer_durations = profile_model_layers_with_strategy(model, encoded_inputs, strategy)

    # Print the average duration of each layer
    # for i, duration in enumerate(average_layer_durations):
    #     print(f"Layer {i + 1} Average Duration: {duration:.4f} seconds")

    # Assuming 1 unit of power consumption
    power_consumption = 1.0
    average_edp_values = [duration * power_consumption for duration in average_layer_durations]

    return average_layer_durations, average_edp_values