import csv

path = "native_runner/probe/out/engine-cluster-apk/source/assets/csv_logic/game_modes.csv"
with open(path, "r", encoding="utf-8", errors="replace") as f:
    reader = list(csv.reader(f))

headers = reader[0]
print("Total rows:", len(reader))
for idx, row in enumerate(reader):
    line_str = ",".join(row)
    if "7x" in line_str.lower() or "seven" in line_str.lower() or "infinit" in line_str.lower():
        print(f"Match at row {idx}: {row[:5]}")
        for h, v in zip(headers, row):
            if v:
                print(f"  {h} = {v}")
