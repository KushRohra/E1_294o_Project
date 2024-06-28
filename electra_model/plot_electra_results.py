from datasets import load_dataset
from electra_baseline import electra_baseline
from electra_improved_edp import electra_improved_edp
from util import make_plot


def plot_electra_results(average_layer_durations, average_edp_values, average_layer_durations_improved, average_edp_values_improved):
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
    dataset = load_dataset('ag_news', split='test[:20]')
    test_dataset = [item['text'] for item in dataset]

    average_baseline_duration, average_layer_durations, average_edp_values = electra_baseline(test_dataset)
    average_layer_durations_improved, average_edp_values_improved = electra_improved_edp(test_dataset)

    layer_duration_percentange_improvements = []
    for i, (average_layer_duration, average_layer_duration_improved) in enumerate(zip(average_layer_durations, average_layer_durations_improved)):
        percentage_improvement = ((average_layer_duration - average_layer_duration_improved) / average_layer_duration) * 100
        layer_duration_percentange_improvements.append(percentage_improvement)

    print(f"average_layer_durations: {average_layer_durations}")
    print(f"average_layer_durations_improved: {average_layer_durations_improved}")
    print(f"layer_duration_percentange_improvements: {layer_duration_percentange_improvements}")

    plot_electra_results(average_layer_durations, average_edp_values, average_layer_durations_improved, average_edp_values_improved)

'''
average_layer_durations: [0.07371768 0.07123624 0.06948195 0.07011195 0.07185886 0.07414669
 0.07170593 0.06898713 0.07036813 0.06963474 0.06972487 0.06746333
 0.06842051 0.06963035 0.06671692 0.06885062 0.06993268 0.06992246
 0.07025527 0.06939412 0.0680039  0.06815335 0.06955701 0.0696262 ]
average_layer_durations_improved: [0.05913584 0.05919961 0.06009873 0.06322922 0.05957034 0.05891681
 0.05843711 0.05714145 0.05633676 0.05965601 0.05812391 0.05960993
 0.05869894 0.05732939 0.05937353 0.05705374 0.05759473 0.05779173
 0.05877253 0.05878348 0.05972657 0.0603663  0.06009188 0.06317177]
layer_duration_percentange_improvements: [19.780649290648412, 16.896783639284852, 13.504534904628379, 9.816783948494042, 17.100915434380934, 20.54020378223035, 18.504490926129765, 17.17086050602551, 19.939955008608464, 14.33009466076966, 16.63819430030641, 11.640992402858767, 14.208555128575034, 17.66608451918561, 11.006784272152785, 17.134019979538092, 17.64260673280857, 17.348825449606263, 16.344311021120266, 15.290401413728885, 12.171849255887505, 11.42577631348107, 13.607735568860631, 9.27011957020678]
'''