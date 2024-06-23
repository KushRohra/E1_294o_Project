from gpt2_baseline import gpt2_baseline
from gpt2_improved_edp import gpt2_improved_edp
from util import make_plot
from datasets import load_dataset

def plot_gpt_2_results():
    dataset = load_dataset('ag_news', split='test[:20]') 
    test_dataset = [item['text'] for item in dataset]

    average_baseline_duration, average_layer_durations, average_edp_values = gpt2_baseline(test_dataset)
    average_layer_durations_improved, average_edp_values_improved = gpt2_improved_edp(test_dataset)

    epochs = range(1, len(average_layer_durations) + 1)

    make_plot(
        epochs, 
        average_layer_durations, 
        'Baseline Layer Duration', 
        average_layer_durations_improved, 
        'Improved Layer Duration', 
        'Comparison of Average Layer Durations For Pretrained GPT 2 Model', 
        'Epochs', 
        'Average Layer Duration'
    )

    make_plot(
        epochs, 
        average_edp_values, 
        'Baseline EDP', 
        average_edp_values_improved, 
        'Improved EDP', 
        'Comparison of Average EDP Values For Pretrained GPT 2 Model', 
        'Epochs', 
        'Average EDP Value'
    )

if __name__ == "__main__":
    plot_gpt_2_results()