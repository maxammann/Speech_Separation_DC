import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

N_METRICS = 4

def mean_metrics(eval_result, name='SDR'):
    return list(map(lambda source_result: np.nanmean(source_result[name]), eval_result))

def mean_all_metrics(eval_result):
    cases =  map(lambda source_result: "%s - %s" % (source_result['reference_name'], source_result['estimated_name']), eval_result)
    return (
        list(cases), 
        mean_metrics(eval_result, 'SDR'), 
        mean_metrics(eval_result, 'SIR'),
        mean_metrics(eval_result, 'ISR'),
        mean_metrics(eval_result, 'SAR')
    )

def plot_metrics(metrics):
    x = []
    y = []

    y_mean_baseline_sdr = []

    for against_sources, against_rev_sources, baseline in metrics:
        # Decide which matching we want to take (rev or normal)
        #max_index = np.argmax([np.sum(values[1]), np.sum(values_rev[1])])
        # Get best results form (rev or normal)
        #cases, mean_sdrs, mean_sirs = values if max_index == 0 else values_rev

        metric_values = mean_all_metrics(against_sources)
        assert len(metric_values) - 1 == N_METRICS

        x.extend(metric_values[0])
        y.append(metric_values[1:])

        if baseline:
            y_mean_baseline_sdr.extend(mean_metrics(baseline, 'SDR'))

    x = np.arange(len(x))
    y = np.array(y).transpose(1, 0, 2).reshape((N_METRICS, -1))

    mean_mean_y = np.mean(y, axis=1)
    baseline_mean_mean_sir = np.mean(y_mean_baseline_sdr)

    fig, ax = plt.subplots(figsize=(15,4))

    y_mean_sdrs, y_mean_sirs, y_mean_isrs, y_mean_sars = y
    ax.scatter(x, y_mean_sdrs, label='Mean SDRs',s=10)
    ax.scatter(x, y_mean_sirs, label='Mean SIRs',s=10)
    ax.scatter(x, y_mean_isrs, label='Mean ISRs',s=10)
    ax.scatter(x, y_mean_sars, label='Mean SARs',s=10)
    #ax.scatter(np.arange(len(y_mean_baseline_sdr)), y_mean_baseline_sdr, label='Baseline Mean SDRs')

    mean_mean_sdr, mean_mean_sir, mean_mean_isr, mean_mean_sar = mean_mean_y
    #ax.plot(x, [mean_mean_sdr] * len(x), label='Mean-Mean SDR')
    #ax.plot(x, [mean_mean_sir] * len(x), label='Mean-Mean SIR')
    #ax.plot(x, [mean_mean_isr] * len(x), label='Mean-Mean ISR')
    #ax.plot(x, [mean_mean_sar] * len(x), label='Mean-Mean SAR')
    #ax.plot(x, [baseline_mean_mean_sir] * len(x), label='Baseline Mean-Mean SDR')

    print("Mean-Mean SDR:\f%s" % mean_mean_sdr)
    print("Mean-Mean SIR:\f%s" % mean_mean_sir)
    print("Mean-Mean ISR:\f%s" % mean_mean_isr)
    print("Mean-Mean SAR:\f%s" % mean_mean_sar)

    #ax.annotate("%.2f" % mean_mean_sdr, xy=(0, mean_mean_sdr))
    #ax.annotate("%.2f" % baseline_mean_mean_sir, xy=(0, baseline_mean_mean_sir))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=90)
    plt.ylabel('Decibel')
    plt.xlabel('Sample: Reference - Estimated')
    ax.legend(loc='upper right')
    plt.show()