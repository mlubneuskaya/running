import pandas as pd
import numpy as np
import os
import difflib
import re


# --- 1. Functions for Name Extraction and Matching ---


def get_names_from_videos(directory_path):
    """
    Searches a directory for .mov files, strips numbers/underscores,
    and returns a unique list of perfectly formatted names.
    """
    unique_names = set()

    if not os.path.exists(directory_path):
        print(
            f"Warning: The folder '{directory_path}' was not found. Skipping name matching."
        )
        return []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".mov"):
                # 1. Remove the .mov extension
                raw_name = os.path.splitext(file)[0]

                # 2. Chop off the trailing underscore and number (e.g., "_1", "_2")
                clean_name = re.sub(r"_\d+$", "", raw_name)

                # 3. Replace remaining underscores with spaces
                clean_name = clean_name.replace("_", " ")

                # 4. Capitalize it properly (artur brzezinski -> Artur Brzezinski)
                clean_name = clean_name.title()

                unique_names.add(clean_name)

    print(f"Found {len(unique_names)} unique athletes in the video directory.")
    return list(unique_names)


def find_best_match(target_name, valid_names_list):
    """
    Fuzzy matches names. Sorts the words first so 'JUNG Marcin'
    matches 'Marcin Jung' perfectly.
    """
    if pd.isna(target_name) or not valid_names_list:
        return target_name

    def normalize(name):
        return " ".join(sorted(str(name).lower().split()))

    normalized_valid = {normalize(n): n for n in valid_names_list}
    norm_target = normalize(target_name)

    # Cutoff at 0.6 allows for slight typos in the CSV export
    matches = difflib.get_close_matches(
        norm_target, normalized_valid.keys(), n=1, cutoff=0.6
    )

    if matches:
        return normalized_valid[matches[0]]
    return target_name


def process_optometrix_csv(input_csv, output_csv, video_directory):
    print(f"Processing {input_csv}...")

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {input_csv}.")
        return

    # Convert any purely empty spaces to actual NaN values immediately
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # --- Step 1: Translate Headers and Deduplicate ---
    translation_dict = {
        "Test": "test",
        "Nazwisko": "surname",
        "Imię": "name",
        "Imię i Nazwisko": "full_name",
        "DataUrodzenia": "birth_date",
        "PBe": "gender",
        "Waga": "weight",
        "Wysoko[": "height",
        "Stopa": "foot",
        "Sport": "sport",
        "Dyscyplina": "discipline",
        "Poziom": "level",
        "Rola": "role",
        "ID": "id",
        "SzkoBa": "school",
        "Numer na koszulce": "jersey_number",
        "Maxymalne tętno": "hr_max",
        "Tętno spoczynkowe": "resting_hr",
        "Anaerobowy tętno maxymalne": "anaerobic_hr_max",
        "Anaerobowe tętno minimalne": "anaerobic_hr_min",
        "Wysoko[ Gyko": "height_gyko",
        "Data": "date",
        "#": "step",
        "Czas[s]": "time",
        "OdlegBo[[cm]": "distance",
        "Czkont.[s]": "contact_time",
        "Czlotu[s]": "flight_time",
        "Faza zamachu[s]": "swing_phase",
        "Stride time[s]": "stride_time",
        "Elewacja[cm]": "elevation",
        "Prędko[[m/s]": "speed",
        "Przysp.[m/s]": "acceleration",
        "Kroki[cm]": "steps",
        "Krok[cm]": "step",
        "Częstotliwo[[krok/s]": "step_frequency_s",
        "Częstotliwo[[krok/m]": "step_frequency_min",
        "Alpha[deg]": "alpha_deg",
        "Nierwnowaga[%]": "imbalance",
        "Podwjne seop.[s]": "double_support",
        "Czasy kroku[s]": "step_time",
        "Duty factor": "duty_factor",
        "Faza kontaktu[s]": "contact_phase",
        "Stopa na pBasko[s]": "foot_flat",
        "Faza napędowa[s]": "propulsive_phase",
        "PCI": "pci",
    }

    new_cols = []
    to_drop = []
    seen = {}
    for col in df.columns:
        new_col_name = translation_dict.get(col, col)

        if "%" in new_col_name:
            # FIX: Append the new column name, not the old one, so df.drop works later!
            to_drop.append(new_col_name)

            # Ensure every column name is mathematically unique
        if new_col_name in seen:
            seen[new_col_name] += 1
            new_col_name = f"{new_col_name}_{seen[new_col_name]}"
        else:
            seen[new_col_name] = 0

        new_cols.append(new_col_name)

    df.columns = new_cols

    # --- Step 2 & 4: Fix structure and remove summaries ---
    if "step" in df.columns:
        step_col_index = np.where(df.columns == "step")[0][0]
        df.iloc[:, :step_col_index] = df.iloc[:, :step_col_index].ffill()
        df = df[df["step"].astype(str).str.strip().str.isdigit()]
    else:
        print("Warning: Could not find the '#' or 'step' column to filter step data.")

    # --- Step 3: Drop entirely empty columns ---
    df = df.dropna(axis=1, how="all")

    # --- Step 5: Fuzzy match names ---
    print("Looking for video files...")
    valid_video_names = get_names_from_videos(video_directory)

    if valid_video_names and "full_name" in df.columns:
        print("Matching names to video files...")
        df["full_name"] = df["full_name"].apply(
            lambda x: find_best_match(x, valid_video_names)
        )

    # --- Step 6: Fix individual Name and Surname ---
    if "full_name" in df.columns:
        print("Fixing individual Name and Surname columns...")
        split_names = df["full_name"].str.split(" ", n=1, expand=True)

        if split_names.shape[1] == 2:
            df["name"] = split_names[0]
            df["surname"] = split_names[1]
        else:
            df["name"] = df["full_name"]
            df["surname"] = ""

    # --- Step 7: Assign study_id and test_id ---
    if "date" in df.columns and "full_name" in df.columns:
        print("Assigning Study IDs and Test IDs...")

        # 1. Convert the string column into Pandas datetime objects
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")

        # 2. study_id: Grab just the YYYY-MM-DD part and rank them globally (Earliest = 1)
        df["study_id"] = parsed_dates.dt.date.rank(method="dense").astype("Int64")

        # 3. test_id: Group by the runner's name, then rank their exact timestamps chronologically
        df["test_id"] = (
            parsed_dates.groupby(df["full_name"]).rank(method="dense").astype("Int64")
        )

        # Save the cleaned data
    df.drop(columns=to_drop, errors="ignore").to_csv(
        output_csv, index=False, encoding="utf-8-sig"
    )
    print(f"Success! Cleaned data saved to {output_csv}")


# --- How to use ---
if __name__ == "__main__":
    # 1. Ensure your input CSV matches the name of the file outputted by your XML script
    INPUT_CSV = "../data/input/optojump/optojump_output/raw/optojump_basic.csv"

    # 2. This is what the final, perfect file will be called
    OUTPUT_CSV = "../data/input/optojump/optojump_output/parsed/optojump_basic.csv"

    # 3. Put your .mov files in a folder called 'videos' in the same directory as this script,
    # or change this path to wherever they currently live.
    VIDEO_DIR = "../data/input/optojump"

    process_optometrix_csv(INPUT_CSV, OUTPUT_CSV, VIDEO_DIR)
