from transformers import TFBertForSequenceClassification, BertTokenizer
import time
import matplotlib.pyplot as plt
import os

path = os.path.join(os.getcwd(), "bert_model")


def make_plot(epochs, plot_1_values, label_1, plot_2_values, label_2, title, x_label, y_label):
    plt.figure(figsize=(10, 6))

    # Plotting average baseline duration
    plt.plot(epochs, plot_1_values, label=label_1)

    # Plotting improved average layer durations
    plt.plot(epochs, plot_2_values, label=label_2)

    # Adding labels and title
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
        # Forward pass through the model
        model(encoded_input)
        end_time = time.time()
        # Calculate the duration
        durations.append(end_time - start_time)
    return durations


def get_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    return model, tokenizer
