from datasets import load_dataset

from bert_baseline import bert_baseline
from bert_improved_edp import bert_improved_edp
from util import make_plot


def plot_bert_results(average_layer_durations, average_edp_values, average_layer_durations_improved, average_edp_values_improved):
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
    dataset = load_dataset('ag_news', split='test[:100]')
    test_dataset = [item['text'] for item in dataset]

    average_baseline_duration, average_layer_durations, average_edp_values = bert_baseline(test_dataset)
    average_layer_durations_improved, average_edp_values_improved = bert_improved_edp(test_dataset)

    layer_duration_percentange_improvements = []
    for i, (average_layer_duration, average_layer_duration_improved) in enumerate(zip(average_layer_durations, average_layer_durations_improved)):
        percentage_improvement = ((average_layer_duration - average_layer_duration_improved) / average_layer_duration) * 100
        layer_duration_percentange_improvements.append(percentage_improvement)

    print(f"average_layer_durations: {average_layer_durations}")
    print(f"average_layer_durations_improved: {average_layer_durations_improved}")
    print(f"layer_duration_percentange_improvements: {layer_duration_percentange_improvements}")

    plot_bert_results(average_layer_durations, average_edp_values, average_layer_durations_improved, average_edp_values_improved)

'''
average_layer_durations: [0.03737931 0.03657056 0.03609989 0.03648033 0.03673519 0.03710644
 0.03752869 0.03597507 0.0361626  0.03708625 0.03793008 0.03693984]
average_layer_durations_improved: [0.03633713 0.0355455  0.03575294 0.03522357 0.03483485 0.03505436
 0.03486017 0.03530635 0.03530423 0.03564917 0.03594803 0.03556232]
layer_duration_percentange_improvements: [2.7881075273672873, 2.802975851117472, 0.9610873293551881, 3.4450452782047662, 5.173087990107901, 5.530255016073428, 7.11060403131402, 1.8588526413113624, 2.3736500503207627, 3.8749668034942473, 5.225523848818654, 3.729078905144107]
'''