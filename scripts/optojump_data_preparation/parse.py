import xml.etree.ElementTree as ET
import csv
import re


def convert_excel_xml_to_csv(input_xml_path, output_csv_path):
    print(f"Reading {input_xml_path}...")

    try:
        with open(input_xml_path, "r", encoding="utf-8", errors="ignore") as file:
            xml_string = file.read()

        xml_string = xml_string.replace("\x19", "ę")
        xml_string = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", xml_string)

        root = ET.fromstring(xml_string)

    except Exception as e:
        print(f"Error parsing file: {e}")
        return

    namespaces = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    ss_uri = namespaces["ss"]

    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        row_count = 0

        for row in root.findall(".//ss:Worksheet/ss:Table/ss:Row", namespaces):
            row_data = []
            current_col_index = 1

            for cell in row.findall("ss:Cell", namespaces):
                index_attr = cell.get(f"{{{ss_uri}}}Index")

                if index_attr is not None:
                    target_col = int(index_attr)
                    while current_col_index < target_col:
                        row_data.append("")
                        current_col_index += 1

                data_tag = cell.find("ss:Data", namespaces)
                if data_tag is not None and data_tag.text is not None:
                    row_data.append(data_tag.text.strip())
                else:
                    row_data.append("")

                current_col_index += 1

            if any(row_data):
                writer.writerow(row_data)
                row_count += 1

    print(f"Success! {row_count} rows written to {output_csv_path}")


if __name__ == "__main__":
    INPUT_FILE = "data/input/optojump/optojump_output/raw/optojump_basic.xml"
    OUTPUT_FILE = "data/input/optojump/optojump_output/raw/optojump_basic.csv"

    convert_excel_xml_to_csv(INPUT_FILE, OUTPUT_FILE)
