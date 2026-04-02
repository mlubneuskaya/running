import os
import re
import unicodedata


def clean_polish_chars(text):
    text = text.replace("ł", "l").replace("Ł", "L")
    temp_str = unicodedata.normalize("NFD", text)
    return "".join([c for c in temp_str if not unicodedata.combining(c)]).lower()


def rename_files(directory_path):
    pattern1 = re.compile(r"^(\S+)\s+(\S+).*\s+(\d+):\d+\.mov$", re.IGNORECASE)

    pattern2 = re.compile(r"^([^_]+)_([^_]+)_(\d+)\.mov$", re.IGNORECASE)

    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        match1 = pattern1.match(filename)
        match2 = pattern2.match(filename)

        if match1:
            name = clean_polish_chars(match1.group(1))
            surname = clean_polish_chars(match1.group(2))
            file_num = match1.group(3)
            new_name = f"{name}_{surname}_{file_num}.mov"

        elif match2:
            name = clean_polish_chars(match2.group(1))
            surname = clean_polish_chars(match2.group(2))
            file_num = match2.group(3)
            new_name = f"{name}_{surname}_{file_num}.mov"

            if filename == new_name:
                print(f"Skipped: '{filename}' (Already perfectly formatted)")
                continue

        else:
            print(f"Skipped: '{filename}' (Pattern did not match)")
            continue

        old_file = os.path.join(directory_path, filename)
        new_file = os.path.join(directory_path, new_name)

        try:
            os.rename(old_file, new_file)
            print(f"Renamed: '{filename}' -> '{new_name}'")
        except OSError as e:
            print(f"Error renaming {filename}: {e}")


if __name__ == '__main__':
    path = "../data/input/optojump/study_1"
    rename_files(path)
