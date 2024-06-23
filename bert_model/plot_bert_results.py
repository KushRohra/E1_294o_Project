from datasets import load_dataset

from bert_baseline import bert_baseline
from bert_improved_edp import bert_improved_edp
from util import make_plot


def plot_bert_results():
    # test_dataset = [
    #     "Hello, my name is ChatGPT. How can I assist you today?",
    #     "What is the capital of France?",
    #     "Explain the theory of relativity.",
    #     "How do I bake a chocolate cake?",
    #     "What are the benefits of machine learning?"
    # ]

    dataset = load_dataset('ag_news', split='test[:20]')
    test_dataset = [item['text'] for item in dataset]

    average_baseline_duration, average_layer_durations, average_edp_values = bert_baseline(test_dataset)
    average_layer_durations_improved, average_edp_values_improved = bert_improved_edp(test_dataset)

    epochs = range(1, len(average_layer_durations) + 1)

    make_plot(
        epochs,
        average_layer_durations,
        'Baseline Layer Duration',
        average_layer_durations_improved,
        'Improved Layer Duration',
        'Comparison of Average Layer Durations For Pretrained BERT Model',
        'Epochs',
        'Average Layer Duration'
    )

    make_plot(
        epochs,
        average_edp_values,
        'Baseline EDP',
        average_edp_values_improved,
        'Improved EDP',
        'Comparison of Average EDP Values For Pretrained BERT Model',
        'Epochs',
        'Average EDP Value'
    )


if __name__ == "__main__":
    plot_bert_results()
