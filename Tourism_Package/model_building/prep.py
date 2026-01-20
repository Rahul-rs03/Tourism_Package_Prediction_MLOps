
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from datasets import load_dataset

api = HfApi()
dataset = load_dataset("rahulsuren12/tourism-package-prediction")
df = dataset["train"].to_pandas()


#-----------------------
#Drop Columns
#-----------------------
cols_to_drop = ["Unnamed: 0", "CustomerID"]

existing_cols = [col for col in cols_to_drop if col in df.columns]

if existing_cols:
    df.drop(columns=existing_cols, inplace=True)

#-----------------------
# Remove duplicates
#-----------------------
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    df = df.drop_duplicates()

#-----------------------
# Treating error in Gender Column
#-----------------------
df.Gender = df.Gender.replace("Fe Male","Female")

#-----------------------
# Transform CityTier column
#-----------------------
if "CityTier" in df.columns:
    df["CityTier"] = df["CityTier"].map({
        1: "Tier1",
        2: "Tier2",
        3: "Tier3"
    })

# Explicitly set dtype to object
df["CityTier"] = df["CityTier"].astype("object")

#-----------------------
# Split the cleaned dataset into training and testing sets
#-----------------------
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

#-----------------------
# Perform train-test split
#-----------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]


for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"processed/{file_path}",
        repo_id="rahulsuren12/tourism-package-prediction",
        repo_type="dataset",
    )

print("Processed datasets uploaded successfully under /processed folder.")
