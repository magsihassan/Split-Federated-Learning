import json
import os
import matplotlib.pyplot as plt


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_curve(sl, gsfl, metric, ylabel, filename):
    rounds = list(range(1, len(sl[metric]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, sl[metric], label="SL", marker="o")
    plt.plot(rounds, gsfl[metric], label="GSFL", marker="s")
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison (SL vs GSFL)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("results", filename))
    plt.close()


def main():
    sl_path = "results/sl_results.json"
    gsfl_path = "results/gsfl_results.json"

    if not os.path.exists(sl_path) or not os.path.exists(gsfl_path):
        print("❌ Error: Run run_sl.py and run_gsfl.py first!")
        return

    sl = load_results(sl_path)
    gsfl = load_results(gsfl_path)

    os.makedirs("results", exist_ok=True)

    print("🖼 Generating plots...")

    plot_curve(sl, gsfl, "accuracy", "Accuracy", "sl_vs_gsfl_accuracy.png")
    plot_curve(sl, gsfl, "loss", "Loss", "sl_vs_gsfl_loss.png")
    plot_curve(sl, gsfl, "uplink", "Uplink Time (s)", "sl_vs_gsfl_uplink.png")
    plot_curve(sl, gsfl, "downlink", "Downlink Time (s)", "sl_vs_gsfl_downlink.png")
    plot_curve(sl, gsfl, "compute", "Compute Time (s)", "sl_vs_gsfl_compute.png")

    print("🎉 Plots generated inside /results folder!")


if __name__ == "__main__":
    main()
