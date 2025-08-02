# trade date downloaded from https://www.bybit.com/derivatives/en/history-data

import pandas as pd
import glob
import os

def combine_csv_files(file_pattern, output_filename):
    """
    Combines multiple CSV files matching a pattern into a single DataFrame
    and writes it to a new CSV file.

    Args:
        file_pattern (str): A glob pattern to find the input CSV files
                            (e.g., "BTCUSDT_*.csv").
        output_filename (str): The name of the CSV file to save the combined data.
    """
    # Find all files matching the pattern
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        print(f"No files found matching the pattern: {file_pattern}")
        return

    # Sort files to ensure a consistent order if needed (e.g., by date in filename)
    # This step is good practice, especially if your filenames have a natural order.
    csv_files.sort()
    print(f"Found {len(csv_files)} files to combine:")
    for f_name in csv_files:
        print(f"  - {f_name}")

    # List to hold DataFrames
    df_list = []

    # Loop through the files and read them into DataFrames
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"Successfully read and added {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
            # Optionally, you might want to skip the file or stop the process
            # For now, we'll just print an error and continue

    if not df_list:
        print("No dataframes were successfully read. Exiting.")
        return

    # Concatenate all DataFrames in the list
    # ignore_index=True will create a new continuous index for the combined DataFrame.
    # If your 'id' column is already globally unique or you want to preserve the original
    # indices from each file (which might lead to duplicate index values), set ignore_index=False.
    # Given your 'id' seems to restart, ignore_index=False is probably what you want if 'id'
    # from the original files is important as is. If you want a new unique index for the
    # combined file, use ignore_index=True. Let's assume you want to keep original 'id'.
    combined_df = pd.concat(df_list, ignore_index=False) # Keep original 'id's and indices
    
    # Optional: If you want a new, unique, continuous index for the combined DataFrame, use:
    # combined_df = pd.concat(df_list, ignore_index=True)
    # This would discard the original DataFrame indices and create a new one from 0 to N-1.
    # The 'id' column from your files would still be present.

    # Optional: Sort the combined DataFrame by timestamp if it's not already guaranteed
    if 'timestamp' in combined_df.columns:
        print("Sorting combined data by 'timestamp'...")
        combined_df.sort_values(by='timestamp', inplace=True)
        # If you used ignore_index=True above and want the index to reflect the new sort order:
        # combined_df.reset_index(drop=True, inplace=True)
    else:
        print("Warning: 'timestamp' column not found for sorting.")


    # Write the combined DataFrame to a new CSV file
    # index=False prevents pandas from writing the DataFrame index as a column
    try:
        combined_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully combined {len(df_list)} files into {output_filename}")
        print(f"Total rows in combined file: {len(combined_df)}")
    except Exception as e:
        print(f"Error writing combined data to {output_filename}: {e}")

# --- How to use ---
if __name__ == "__main__":
    # Define the pattern for your input files
    # This will match files like BTCUSDT_2025-05-01.csv, BTCUSDT_2025-05-02.csv, etc.
    input_file_pattern = "../data/BTCUSDT_*.csv" 
    
    # More specific pattern if you only want files from a certain year/month:
    # input_file_pattern = "BTCUSDT_2025-05-*.csv" 

    # Define the name for your output file
    output_file = "BTCUSDT_combined_all_days.csv"

    # Call the function
    combine_csv_files(input_file_pattern, output_file)

    # You can then load and inspect the combined file:
    # try:
    #     final_df = pd.read_csv(output_file)
    #     print("\nFirst 5 rows of the combined file:")
    #     print(final_df.head())
    #     print("\nLast 5 rows of the combined file:")
    #     print(final_df.tail())
    #     print(f"\nInfo about the combined DataFrame:")
    #     final_df.info()
    # except FileNotFoundError:
    #     print(f"\nOutput file {output_file} not found. Check for errors above.")