from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import time
import matplotlib.pyplot as plt
import os
import tensorflow as tf

path = os.path.join(os.getcwd(), "t5_model")


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


def profile_model(model, encoded_inputs):
    durations = []
    for encoded_input in encoded_inputs:
        start_time = time.time()
        attention_mask = tf.cast(tf.ones_like(encoded_input['input_ids']), dtype=tf.float32)
        decoder_attention_mask = tf.cast(tf.ones_like(encoded_input['input_ids']), dtype=tf.float32)
        model(input_ids=encoded_input['input_ids'], attention_mask=attention_mask,
              decoder_input_ids=encoded_input['input_ids'], decoder_attention_mask=decoder_attention_mask)
        end_time = time.time()
        durations.append(end_time - start_time)
    return durations


def get_model_and_tokenizer():
    # TODO: Check against different variants of T5 models
    model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer
