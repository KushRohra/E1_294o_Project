from datasets import load_dataset
from electra_baseline import electra_baseline
from electra_improved_edp import electra_improved_edp
from util import make_plot


def plot_electra_results():
    dataset = load_dataset('ag_news', split='test[:20]')
    test_dataset = [item['text'] for item in dataset]

    average_baseline_duration, average_layer_durations, average_edp_values = electra_baseline(test_dataset)
    average_layer_durations_improved, average_edp_values_improved = electra_improved_edp(test_dataset)

    epochs = range(1, len(average_layer_durations) + 1)

    make_plot(
        epochs,
        average_layer_durations,
        'Baseline Layer Duration',
        average_layer_durations_improved,
        'Improved Layer Duration',
        'Comparison of Average Layer Durations For Pretrained Electra Model',
        'Epochs',
        'Average Layer Duration'
    )

    make_plot(
        epochs,
        average_edp_values,
        'Baseline EDP',
        average_edp_values_improved,
        'Improved EDP',
        'Comparison of Average EDP Values For Pretrained Electra Model',
        'Epochs',
        'Average EDP Value'
    )


if __name__ == "__main__":
    plot_electra_results()
