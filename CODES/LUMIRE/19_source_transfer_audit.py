# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import csv


SOURCE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA"

FILES = [
    "17_hbgd_growth_parameters.csv",
    "17_hbgd_mathematical_features.csv",
    "17_hbgd_state_parameters.csv",
    "17_hbgd_volume_parameters.csv",
    "17_hbgd_diameter_trajectories.csv",
    "21_individualized_hbgd_predictions.csv",
    "21_individualized_hbgd_patient_performance.csv"
]


def read_csv(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


print("=" * 70)
print("STEP 19 SOURCE DATA AUDIT")
print("=" * 70)


for filename in FILES:

    path = os.path.join(
        SOURCE_DIR,
        filename
    )

    print("\n" + "-" * 70)

    print(filename)

    if not os.path.exists(path):

        print("NOT FOUND")

        continue

    rows = read_csv(path)

    print(
        "Rows:",
        len(rows)
    )

    if not rows:
        continue

    print(
        "Columns:"
    )

    for column in rows[0].keys():

        print(
            "  " + column
        )

    if "patient_id" in rows[0]:

        patients = sorted(
            list(
                set(
                    str(
                        row[
                            "patient_id"
                        ]
                    )

                    for row in rows
                )
            )
        )

        print(
            "Unique patients:",
            len(patients)
        )

        print(
            "Patients:"
        )

        print(
            patients
        )

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)