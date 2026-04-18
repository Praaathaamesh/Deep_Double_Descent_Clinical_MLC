from mcc_epoch_doubdesc_pipeline import run_experiment, plot_double_descent
def main(): 
    # propagate the trail run
    history = run_experiment(noise_rate=0.15, width=64, subset_size=3000, epochs=400)
    print(history)

    # plot the epoch dd mcc graph
    plot_double_descent(history, title_suffix='(noise=15%, width=64)')

if __name__ == "__main__":
    main()