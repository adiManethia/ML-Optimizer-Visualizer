import os
import matplotlib.pyplot as plt

# ----------------------
# Output directory
# ----------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------
# 1. Combined loss plot (log scale)
# ----------------------
def plot_loss_log_scale(histories, save=True):
    plt.figure()
    for name, history in histories.items():
        plt.plot(history["loss"], label=name)

    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("Loss vs Iterations (Log Scale)")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(
            f"{OUTPUT_DIR}/loss_log_scale.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ----------------------
# 2. Separate loss plots (clarity)
# ----------------------
def plot_loss_separately(histories, save=True):
    for name, history in histories.items():
        plt.figure()
        plt.plot(history["loss"])

        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"{name} - Loss vs Iterations")
        plt.grid(True)

        if save:
            plt.savefig(
                f"{OUTPUT_DIR}/loss_{name.lower()}.png",
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()


# ----------------------
# 3. Parameter convergence
# ----------------------
def plot_parameters(histories, save=True):
    # ---- w convergence ----
    plt.figure()
    for name, history in histories.items():
        plt.plot(history["w"], label=name)

    plt.xlabel("Iterations")
    plt.ylabel("w value")
    plt.title("Weight (w) Convergence")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(
            f"{OUTPUT_DIR}/w_convergence.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    # ---- b convergence ----
    plt.figure()
    for name, history in histories.items():
        plt.plot(history["b"], label=name)

    plt.xlabel("Iterations")
    plt.ylabel("b value")
    plt.title("Bias (b) Convergence")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(
            f"{OUTPUT_DIR}/b_convergence.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ----------------------
# 4. Quantitative comparison
# ----------------------
def print_optimizer_summary(histories, threshold=0.05):
    print("\nOptimizer Comparison Summary")
    print("-" * 50)
    print(f"{'Optimizer':10} | {'Final Loss':12} | Iters < {threshold}")
    print("-" * 50)

    for name, history in histories.items():
        final_loss = history["loss"][-1]

        iters = next(
            (i for i, loss in enumerate(history["loss"]) if loss < threshold),
            None
        )

        iters_str = iters if iters is not None else "Not reached"

        print(
            f"{name:10} | {final_loss:<12.6f} | {iters_str}"
        )
