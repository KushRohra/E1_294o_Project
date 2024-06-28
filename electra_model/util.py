import tensorflow as tf
import time
import os
from transformers import TFElectraForSequenceClassification, ElectraTokenizer
import matplotlib.pyplot as plt

path = os.getcwd()


def get_model_and_tokenizer():
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator')
    model = TFElectraForSequenceClassification.from_pretrained('google/electra-large-discriminator')
    return model, tokenizer


def profile_model_layers(model, encoded_inputs):
    transformer_layers = model.electra.encoder.layer
    layer_times = []

    for encoded_input in encoded_inputs:
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input.get('attention_mask', tf.ones_like(input_ids))
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)

        outputs = model.electra(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

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
                output_attentions=False
            )
            end_time = time.time()
            input_layer_times.append(end_time - start_time)
            hidden_states = layer_output[0]
        layer_times.append(input_layer_times)

    return layer_times


def profile_model(model, encoded_inputs):
    durations = []
    for encoded_input in encoded_inputs:
        start_time = time.time()
        model(encoded_input)
        end_time = time.time()
        durations.append(end_time - start_time)
    return durations


def make_plot(epochs, plot_1_values, label_1, plot_2_values, label_2, title, x_label, y_label):
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, plot_1_values, label=label_1)
    plt.plot(epochs, plot_2_values, label=label_2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(path, f"{title}.png"))
