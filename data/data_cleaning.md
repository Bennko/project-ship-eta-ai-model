## Data Analysis and Cleaning

For the data cleaning of both files (1. Felixstowe -> Rotterdam, 2. Rotterdam -> Hamburg) we put together the main thought process here below:

Tools: Pandas, matplotlib

We started by importing both files and triggering a warning on bad lines. (lines where there might be potential errors).

1. Checking for duplicates
We used the drop_duplicates() function to remove all rows which have the exact same entry for in every column. It was observed that multiple rows were duplicated multiple times. 

2. Distribution of Data
We then wanted to get an overview of the distribution of values in each column as well as their saved data types. 

3. Handling missing values
Next, we also checked for missing values in both datasets. We first started by using the pandas function isnull() which checks if a value in a given row/colum is null. This did not prove useful as we found out that the missing values in both datasets were represented by the value '?'. We replaced all the missing values with the most occoring entry in the trip, since the columns where missing data is are all static.
We couldn't apply the same technique to somne columns such as: Draught and AisSourcen, since in the trips where the data is missing, besides '?' there was no further data to fill up the missing data.

4. Cross-checking the validity of data
Here we wanted to check whether the data we have is physically possible(matches the route) and meets the AIS data standards in order to identify outliers if present. This was done by considering if:
    i. Latitude and Longitude ranges are valid (-90 to 90 for latitude, -180 to 180 for longitude)
    ii. The start time and end time reasonable.
    iii. The position of the ship within reasonable locations of the trip for both routes.

For the trip Felixstowe - Hamburg it was observed that there were entries for Latitudes < 51.2 and Latitudes > 52.2 which were visual outliers after plotting all the routes of all trips. These were removed from the dataset.

For the trip Rotterdam - Hamburg, no such outliers were observed.


5. Addition of new features
we added some new features that could be interesting for further analysis such as:
    1.  pastTravelTime: describes the already traveled time in  seconds.
    2.  timeTillArrival: describes the time until the vessel reaches its destination, this is also our target feature


6. Deletion of extremes
Whilst checking for the distribution of values in the dataset, we included a check for the distribution of trip durations which we plotted as a graph (see Rotterdam-Hamburg for reference). We then considered and decided to eleminate the extremes of the normal distribution observed in the graph by cutting off trips that were either too long or short.
