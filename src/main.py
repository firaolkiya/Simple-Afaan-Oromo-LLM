import pandas as pd


df = pd.read_csv("data/data.csv")

print(df.shape)
print(df.columns)

with open("data/oromic_data.txt","w") as file:
    for i in range(1919):
        file.write(df["Information"].iloc[i]+"\n")
file.close()
with open("data/obn data.txt") as file:
    with open("data/oromic_data.txt","w") as target_file:
        for line in file.readlines():
            if line.strip():
                new_line = line.strip()
                new_line = new_line.strip("\n")
                target_file.write(new_line+"\n")
        target_file.close()
