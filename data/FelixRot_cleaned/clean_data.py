import pandas as pd

def interpolate_columns(df, target_columns=['Destination', 'Name', 'Callsign']):

    last_values = {col: None for col in target_columns}

    for index, row in df.iterrows():
        trip_id = row['TripID']
        for col in target_columns:
            if row[col] != '?':
                last_values[col] = row[col]
            elif row[col] == '?' and last_values[col] is not None:
                df.at[index, col] = last_values[col]

    return df

# Read file
df = pd.read_csv("felixstowe_rotterdam.csv", on_bad_lines='warn')

# drop duplicates
no_duplicates = df.drop_duplicates()

# replace questionmarks
no_duplicates["AisSourcen"].replace({'?': 'H7001'}, inplace = True)

filtered_mmsis = df[df['MMSI'].astype(str).str.len() == 9]['MMSI']
sorted_rows = no_duplicates.sort_values(by=['TripID', 'MMSI', 'Destination', 'Callsign'])


sorted_rows2 = sorted_rows

clean_df = interpolate_columns(sorted_rows2)

clean_df.drop(clean_df.loc[clean_df['Destination']== 'FELIXSTOWE'].index, inplace=True)

clean_df2 = clean_df[clean_df['Latitude'] >= 51.2] 

clean_df3 = clean_df[clean_df['Longitude'] <= 52.2]

clean_df3 = clean_df3.drop_duplicates()

clean_df3.insert(0, "pastTravelTime",(pd.to_datetime(clean_df3["time"]) - pd.to_datetime(clean_df3["StartTime"])).dt.total_seconds())
clean_df3.insert(0, "timeTillArrival", (pd.to_datetime(clean_df3["EndTime"]) - pd.to_datetime(clean_df3["time"])).dt.total_seconds())

clean_df3.to_csv('felixstowe_rotterdam_clean_new.csv', index=False)