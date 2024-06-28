from roberta_baseline import roberta_baseline
from roberta_improved_edp import roberta_improved_edp
from util import make_plot
from datasets import load_dataset

def plot_roberta_results(average_layer_durations, average_edp_values, average_layer_durations_improved, average_edp_values_improved):
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
    dataset = load_dataset('ag_news', split='test[:500]') 
    test_dataset = [item['text'] for item in dataset]

    average_baseline_duration, average_layer_durations, average_edp_values = roberta_baseline(test_dataset)
    average_layer_durations_improved, average_edp_values_improved = roberta_improved_edp(test_dataset)

    layer_duration_percentange_improvements = []
    for i, (average_layer_duration, average_layer_duration_improved) in enumerate(zip(average_layer_durations, average_layer_durations_improved)):
        percentage_improvement = ((average_layer_duration - average_layer_duration_improved) / average_layer_duration) * 100
        layer_duration_percentange_improvements.append(percentage_improvement)

    print(f"average_layer_durations: {average_layer_durations}")
    print(f"average_layer_durations_improved: {average_layer_durations_improved}")
    print(f"layer_duration_percentange_improvements: {layer_duration_percentange_improvements}")

    plot_roberta_results(average_layer_durations, average_edp_values, average_layer_durations_improved, average_edp_values_improved)

'''
average_layer_durations: [0.02437332 0.02288621 0.02295941 0.02330185 0.02378273 0.02370207
 0.023613   0.02428863 0.02355782 0.02405931 0.02349113 0.02386618]
average_layer_durations_improved: [0.03012606 0.02904337 0.02879929 0.02861938 0.02842992 0.02946862
 0.02898572 0.02956137 0.02939479 0.0294427  0.02909209 0.02826433]
layer_duration_percentange_improvements: [-23.602589132811975, -26.903335397886284, -25.43566353130553, -22.82022507755715, -19.540205664067443, -24.32932423408986, -22.75319629068286, -21.708652688916295, -24.777219940955778, -22.37550131762233, -23.842877994744615, -18.428361174526394]
'''