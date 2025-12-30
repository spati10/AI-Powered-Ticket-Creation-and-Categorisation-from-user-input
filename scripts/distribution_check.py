import pandas as pd

df = pd.read_csv("data/IT Support Ticket Data.csv")
print("\nCategory distribution:\n")
print(df["Category"].value_counts())

print("\nPriority distribution:\n")
print(df["Priority"].value_counts())
