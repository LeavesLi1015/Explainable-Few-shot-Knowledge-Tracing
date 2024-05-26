# read the last line of all the file ended wiht "_predict.txt"
# convert the last line to dictionary
# print the file name and dictionary["windowacclate_all"],dictionary["windowprecisionlate_all"],dictionary["windowrecalllate_all"],dictionary["windowf1late_all"]

import os

for file in os.listdir():
    if file.endswith("_predict.txt") and file.startswith("2_"):
        with open(file, 'r') as f:
            last_line = f.readlines()[-1]
            dictionary = eval(last_line)
            print(file)

            print(f"windowacclate_all: {dictionary['windowacclate_all']}")
            print(f"windowprecisionlate_all: {dictionary['windowprecisionlate_all']}")
            print(f"windowrecalllate_all: {dictionary['windowrecalllate_all']}")
            print(f"windowf1late_all: {dictionary['windowf1late_all']}")