from mlc_epoch_doubdesc_pipeline import run_experiment, plot_double_descent

if __name__ == "__main__":
    
    # propagate the trail run
    history = run_experiment(noise_rate=0.15, width=64, subset_size=3000, epochs=400)
    print(history)

    # plot the epoch dd mlc graph
    plot_double_descent(history, title_suffix='(noise=15%, width=64)')