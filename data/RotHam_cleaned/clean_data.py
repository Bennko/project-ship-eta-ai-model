import pandas as pd

def replaceMissing(column):
    for tripId in df_no_duplicates["TripID"].unique():
        tripSelectedColumn = df_no_duplicates.loc[df_no_duplicates["TripID"] == tripId, column]
        if tripSelectedColumn.value_counts().idxmax() == "?" and len(tripSelectedColumn.value_counts()) > 1:
            new_value = tripSelectedColumn.value_counts().sort_values(ascending=False).index[1]
        else:
            new_value = tripSelectedColumn.value_counts().idxmax()

        df_no_duplicates.loc[df_no_duplicates["TripID"] == tripId, column] = new_value # type: ignore
        
# Load csv file into dataframe
df = pd.read_csv("rotterdam_hamburg.csv", on_bad_lines="warn", dtype={'Draught': str}) 

# drop duplicates
df_no_duplicates = df.drop_duplicates()

# Fill in missing data for "?"
replaceMissing("Destination")  
replaceMissing("Callsign")
replaceMissing("Name")

# Drop columns where filling data this is not possible
df_no_duplicates.drop(columns=["Draught", "AisSourcen"], inplace=True)  

# Standardize the data
df_no_duplicates.loc[: , "Destination"] = "HAMBURG"

#Add two new features which are timeTillArrival and pastTravelTime
df_no_duplicates.insert(0, "pastTravelTime",(pd.to_datetime(df_no_duplicates["time"]) - pd.to_datetime(df_no_duplicates["StartTime"])).dt.total_seconds())
df_no_duplicates.insert(0, "timeTillArrival", (pd.to_datetime(df_no_duplicates["EndTime"]) - pd.to_datetime(df_no_duplicates["time"])).dt.total_seconds())

# drop unrealistic values
df_no_duplicates["StartTime"] = pd.to_datetime(df_no_duplicates["StartTime"])
df_no_duplicates["EndTime"] = pd.to_datetime(df_no_duplicates["EndTime"])
trip_lenght_seconds = (df_no_duplicates["EndTime"] - df_no_duplicates["StartTime"]).dt.total_seconds()
df_no_duplicates = df_no_duplicates[(trip_lenght_seconds > 65000) & (trip_lenght_seconds < 87500) ]

# Write cleaned data into CSV-file
try:
    df_no_duplicates.to_csv('rotterdam_hamburg_clean_test.csv', index=False)
    print("CSV file was successfully created. The cleaned dataset has been saved to rotterdam_hamburg_clean.csv in the same folder!")
except Exception as e:
    print("An error occurred:", e)
