from t5_baseline import t5_baseline
from t5_improved_edp import t5_improved_edp
from util import make_plot
from datasets import load_dataset

def plot_t5_results(average_layer_durations, average_edp_values, average_layer_durations_improved, average_edp_values_improved):
    epochs = range(1, len(average_layer_durations) + 1)

    make_plot(
        epochs,
        average_layer_durations,
        'Baseline Layer Duration',
        average_layer_durations_improved,
        'Improved Layer Duration',
        'Comparison of Average Layer Durations For Pretrained T5 Model',
        'Epochs',
        'Average Layer Duration'
    )

    make_plot(
        epochs,
        average_edp_values,
        'Baseline EDP',
        average_edp_values_improved,
        'Improved EDP',
        'Comparison of Average EDP Values For Pretrained T5 Model',
        'Epochs',
        'Average EDP Value'
    )


if __name__ == "__main__":
    dataset = load_dataset('ag_news', split='test[:20]') 
    test_dataset = [item['text'] for item in dataset]

    average_baseline_duration, average_layer_durations, average_edp_values = t5_baseline(test_dataset)
    average_layer_durations_improved, average_edp_values_improved = t5_improved_edp(test_dataset)

    layer_duration_percentange_improvements = []
    for i, (average_layer_duration, average_layer_duration_improved) in enumerate(zip(average_layer_durations, average_layer_durations_improved)):
        percentage_improvement = ((average_layer_duration - average_layer_duration_improved) / average_layer_duration) * 100
        layer_duration_percentange_improvements.append(percentage_improvement)

    print(f"average_layer_durations: {average_layer_durations}")
    print(f"average_layer_durations_improved: {average_layer_durations_improved}")
    print(f"layer_duration_percentange_improvements: {layer_duration_percentange_improvements}")


    plot_t5_results(average_layer_durations, average_edp_values, average_layer_durations_improved, average_edp_values_improved)

'''
average_layer_durations: [0.02543422 0.02057666 0.02187943 0.02236166 0.02237343 0.02119299]
average_layer_durations_improved: [0.0264721155166626, 0.019745063781738282, 0.019099736213684083, 0.02064269781112671, 0.02062094211578369, 0.020097637176513673]
layer_duration_percentange_improvements: [-4.080705781886475, 4.0414345676222885, 12.704617027536525, 7.687095646091979, 7.832881060986699, 5.168475365860107]
'''