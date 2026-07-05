import pandas as pd

file = "Data/Camera_DetectPeople_Second_Version.xlsx"
output_file = "Camera_DetectPeople_10mn_Version.xlsx"
# อ่านชื่อทุกชีท
excel_file = pd.ExcelFile(file)

print("Sheets found:", excel_file.sheet_names)


def process_sheet(df):
    # สร้างเวลา
    df["time"] = pd.to_datetime(
        df["Hour"].astype(str) + ":" +
        df["Minute"].astype(str) + ":" +
        df["Second"].astype(str),
        format="%H:%M:%S"
    )

    # จัดกลุ่มทุก 10 นาที
    df["time_10min"] = df["time"].dt.floor("10min")

    def aggregate_group(group):
        start_people = group["NumberPeople"].iloc[0]
        total_entry = group["Entry"].sum()
        total_exit = group["Exit"].sum()

        end_people = start_people + total_entry - total_exit

        return pd.Series({
            "Hour": group["time_10min"].iloc[0].hour,
            "Minute": group["time_10min"].iloc[0].minute,
            "Second": 0,
            "Entry": total_entry,
            "Exit": total_exit,
            "NumberPeople": end_people
        })

    result = df.groupby("time_10min").apply(aggregate_group).reset_index(drop=True)

    return result


# เขียนหลายชีทลงไฟล์ใหม่
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    for sheet in excel_file.sheet_names:
        print(f"Processing sheet: {sheet}")

        df = pd.read_excel(file, sheet_name=sheet)

        result = process_sheet(df)

        result.to_excel(writer, sheet_name=sheet, index=False)

print(f"\nDone! Saved to: {output_file}")
