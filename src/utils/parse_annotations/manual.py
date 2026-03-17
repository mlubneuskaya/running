import pandas as pd


def parse_annotations(file_path):
    df = pd.read_csv(file_path, names=["frame", "event_id", "label"], header=0)
    df = df[df["label"] != "Neutral"].copy()
    df['side'] = df['label'].apply(lambda x: 'right' if 'RIGHT' in x.upper() else 'left')

    df['event_type'] = df['label'].apply(
        lambda x: x.split('(')[-1].replace(')', '').strip() if '(' in x else x
    )

    return df