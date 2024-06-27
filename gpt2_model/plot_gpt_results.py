from gpt2_baseline import gpt2_baseline
from gpt2_improved_edp import gpt2_improved_edp
from util import make_plot
from datasets import load_dataset
from datetime import datetime


def plot_gpt_2_results(average_layer_durations, average_edp_values, average_layer_durations_improved, average_edp_values_improved):
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
    print(f"Start time: {datetime.now()}")
    dataset = load_dataset('ag_news', split='test[:500]') 
    test_dataset = [item['text'] for item in dataset]

    average_baseline_duration, average_layer_durations, average_edp_values = gpt2_baseline(test_dataset)
    average_layer_durations_improved, average_edp_values_improved = gpt2_improved_edp(test_dataset)

    layer_duration_percentange_improvements = []
    for i, (average_layer_duration, average_layer_duration_improved) in enumerate(zip(average_layer_durations, average_layer_durations_improved)):
        percentage_improvement = ((average_layer_duration - average_layer_duration_improved) / average_layer_duration) * 100
        layer_duration_percentange_improvements.append(percentage_improvement)

    print(f"average_layer_durations: {average_layer_durations}")
    print(f"average_layer_durations_improved: {average_layer_durations_improved}")
    print(f"layer_duration_percentange_improvements: {layer_duration_percentange_improvements}")

    plot_gpt_2_results(average_layer_durations, average_edp_values, average_layer_durations_improved, average_edp_values_improved)
    print(f"End time: {datetime.now()}")

'''
average_layer_durations: [0.02575966 0.02479245 0.02470555 0.0244479  0.02464197 0.02459989 0.02453771 0.02475476 0.02447546 0.02918374 0.02444964 0.02432861]
average_layer_durations_improved: [0.025700855731964112, 0.02427613115310669, 0.02404357099533081, 0.024003088951110838, 0.023978307247161865, 0.02404534673690796, 0.024104203701019285, 0.024174737453460694, 0.024026403903961182, 0.024016987800598143, 0.02414171552658081, 0.024130097389221193]
layer_duration_percentange_improvements: [0.22827216168073758, 2.0825589982508057, 2.679466245281016, 1.8194121443727358, 2.693223675749727, 2.254244330336835, 1.7666930493253714, 2.3430720974198866, 1.8347237753367238, 17.70421447484915, 1.2594398554294524, 0.8159789572884124]
'''