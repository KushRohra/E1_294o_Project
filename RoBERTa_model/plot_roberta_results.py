from roberta_baseline import roberta_baseline
from roberta_improved_edp import roberta_improved_edp
from util import make_plot

def plot_roberta_results():
    test_dataset = [
        "Hello, my name is ChatGPT. How can I assist you today?",
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "How do I bake a chocolate cake?",
        "What are the benefits of machine learning?"
    ]

    average_baseline_duration, average_layer_durations, average_edp_values = roberta_baseline(test_dataset)
    average_layer_durations_improved, average_edp_values_improved = roberta_improved_edp(test_dataset)

    epochs = range(1, len(average_layer_durations) + 1)

    make_plot(
        epochs, 
        average_layer_durations, 
        'Baseline Layer Duration', 
        average_layer_durations_improved, 
        'Improved Layer Duration', 
        'Comparison of Average Layer Durations For Pretrained RoBERTa Model', 
        'Epochs', 
        'Average Layer Duration'
    )

    make_plot(
        epochs, 
        average_edp_values, 
        'Baseline EDP', 
        average_edp_values_improved, 
        'Improved EDP', 
        'Comparison of Average EDP Values For Pretrained RoBERTa Model', 
        'Epochs', 
        'Average EDP Value'
    )

if __name__ == "__main__":
    plot_roberta_results()