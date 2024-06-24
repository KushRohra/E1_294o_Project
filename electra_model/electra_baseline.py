from util import get_model_and_tokenizer, profile_model_layers, profile_model
import numpy as np


def electra_baseline(test_dataset):
    model, tokenizer = get_model_and_tokenizer()

    encoded_inputs = [tokenizer(text, return_tensors='tf', padding=True, truncation=True) for text in test_dataset]

    baseline_durations = profile_model(model, encoded_inputs)
    average_baseline_duration = np.mean(baseline_durations)

    layer_durations = profile_model_layers(model, encoded_inputs)
    average_layer_durations = np.mean(layer_durations, axis=0)

    power_consumption = 1.0
    average_edp_values = [duration * power_consumption for duration in average_layer_durations]

    return average_baseline_duration, average_layer_durations, average_edp_values