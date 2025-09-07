import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/results.csv")
print(df)

plt.figure(figsize=(6,4))
x = range(len(df))
plt.bar(x, df["clean_acc"], label="Clean")
plt.bar(x, df["noisy_acc"], alpha=0.6, label="Noisy")
plt.xticks(x, df["variant"])
plt.ylabel("Accuracy")
plt.title("Clean vs Noisy Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/robustness_plot.png", dpi=200)
print("Saved plot to outputs/robustness_plot.png")
