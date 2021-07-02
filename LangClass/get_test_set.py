import os
import pandas as df

df = df.DataFrame(columns=["filename", "language"])
os.makedirs('data/common_voice_test/', exist_ok=True)
for root, dirs, files in os.walk('data'):
    for file in files:
        lang = root.split("_")[-1]
        df = df.append({"filename":os.path.join(root, file), "language": lang}, ignore_index=True)
print(df)
test_set = df.sample(300)
print(test_set)

test_set.to_csv("test_set.csv")
test_set_dir = 'data\\common_voice_test\\'
for file in test_set["filename"]:
    root, filename = os.path.split(file)
    lang_dir = os.path.split(root)[1]
    os.makedirs(os.path.join(test_set_dir, lang_dir), exist_ok=True)
    new_path = os.path.join(test_set_dir, lang_dir, filename)
    os.replace(file, new_path)