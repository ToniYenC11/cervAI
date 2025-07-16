import re
import pandas as pd

# Path to the log file
log_path = "retinanet_depth_101_epoch_30_patience_1.txt"

# Read the file
with open(log_path, "r") as f:
    lines = f.readlines()

# Initialize storage
results = []

current_epoch = None
metrics = {}

# Regex patterns
epoch_pattern = re.compile(r"Epoch:\s+(\d+)")
ap_5095_pattern = re.compile(r"Average Precision.*IoU=0.50:0.95.*=\s+([0-9.-]+)")
ap_50_pattern = re.compile(r"Average Precision.*IoU=0.50\s+\|.*=\s+([0-9.-]+)")
ap_75_pattern = re.compile(r"Average Precision.*IoU=0.75\s+\|.*=\s+([0-9.-]+)")
ar_5095_pattern = re.compile(r"Average Recall.*IoU=0.50:0.95.*maxDets=100.*=\s+([0-9.-]+)")

for line in lines:
    epoch_match = epoch_pattern.search(line)
    if epoch_match:
        # When a new epoch is found and metrics from the previous one are complete
        if current_epoch is not None and len(metrics) == 4:
            results.append({
                "Epoch": current_epoch,
                "(AP) @[ IoU=0.50:0.95": metrics["ap_5095"],
                "(AP) @[ IoU=0.50": metrics["ap_50"],
                "(AP) @[ IoU=0.75": metrics["ap_75"],
                "(AR) @[ IoU=0.50:0.95": metrics["ar_5095"],
            })
        # Start collecting new epoch
        current_epoch = int(epoch_match.group(1))
        metrics = {}

    # Match metrics
    if ap_5095_match := ap_5095_pattern.search(line):
        metrics["ap_5095"] = float(ap_5095_match.group(1))
    elif ap_50_match := ap_50_pattern.search(line):
        metrics["ap_50"] = float(ap_50_match.group(1))
    elif ap_75_match := ap_75_pattern.search(line):
        metrics["ap_75"] = float(ap_75_match.group(1))
    elif ar_5095_match := ar_5095_pattern.search(line):
        metrics["ar_5095"] = float(ar_5095_match.group(1))

# Add last epoch's result
if current_epoch is not None and len(metrics) == 4:
    results.append({
        "Epoch": current_epoch,
        "(AP) @[ IoU=0.50:0.95": metrics["ap_5095"],
        "(AP) @[ IoU=0.50": metrics["ap_50"],
        "(AP) @[ IoU=0.75": metrics["ap_75"],
        "(AR) @[ IoU=0.50:0.95": metrics["ar_5095"],
    })

# Convert to DataFrame
df = pd.DataFrame(results)

# Sort by epoch (optional)
df = df.sort_values("Epoch").reset_index(drop=True)

# Save as CSV
save_name = log_path.split(".")[0]

df.to_csv(f"{save_name}.csv", index=False)

print(f"Saved to {save_name}.csv")