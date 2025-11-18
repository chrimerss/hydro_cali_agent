#!/usr/bin/env python3
"""
USGS Gauge Data Download Script

Download discharge data from USGS gauge stations with flexible time resolution options.

Usage:
    python usgs_gauge_download.py --site_num 08165300 --time_start 2025070400 --time_end 2025070523 --time_step 1d
    python usgs_gauge_download.py --site_num 08165300 --time_start 2025070400 --time_end 2025070523 --time_step 1h
    python usgs_gauge_download.py --site_num 08165300 --time_start 2025070400 --time_end 2025070523
"""

import os
import argparse
import pandas as pd
from datetime import datetime
import sys

def parse_time_string(time_str):
    """
    Parse time string in format YYYYMMDDHH to datetime object
    
    Parameters:
    -----------
    time_str : str
        Time string in format YYYYMMDDHH (e.g., '2025070400')
    
    Returns:
    --------
    datetime
        Parsed datetime object
    """
    try:
        if len(time_str) == 10:  # YYYYMMDDHH
            return datetime.strptime(time_str, '%Y%m%d%H')
        elif len(time_str) == 8:  # YYYYMMDD
            return datetime.strptime(time_str, '%Y%m%d')
        else:
            raise ValueError(f"Invalid time format: {time_str}. Expected YYYYMMDDHH or YYYYMMDD")
    except ValueError as e:
        print(f"Error parsing time string '{time_str}': {e}")
        sys.exit(1)

def download_usgs_data(site_code, start_date, end_date, output_dir, time_step='15min'):
    """
    Download discharge data from USGS station and save as CSV file
    
    Parameters:
    -----------
    site_code : str
        USGS station ID
    start_date : datetime
        Start date and time
    end_date : datetime
        End date and time
    output_dir : str
        Output directory path
    time_step : str
        Time resolution: '1d' (daily), '1h' (hourly), '15min' (15-minute, default)
        
    Returns:
    --------
    DataFrame or None
        The downloaded discharge data, or None if download failed
    """
    try:
        import dataretrieval.nwis as nwis
    except ImportError:
        print("Error: dataretrieval package not found. Please install with:")
        print("pip install dataretrieval")
        sys.exit(1)
    
    print(f"Downloading USGS discharge data for station {site_code}")
    print(f"Time period: {start_date} to {end_date}")
    print(f"Time step: {time_step}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Format dates for NWIS service
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Determine service type based on time step
    if time_step == '1d':
        service = 'dv'  # daily values
        stat_code = '00003'  # mean daily discharge
        print("Downloading daily mean discharge data...")
        
        df = nwis.get_record(
            sites=site_code,
            service=service,
            start=start_date_str,
            end=end_date_str,
            parameterCd='00060',  # Discharge parameter
            statCd=stat_code
        )
    else:
        service = 'iv'  # instantaneous values
        print(f"Downloading instantaneous discharge data (resolution: {time_step})...")
        
        df = nwis.get_record(
            sites=site_code,
            service=service,
            start=start_date_str,
            end=end_date_str,
            parameterCd='00060'  # Discharge parameter
        )
        
        # Convert timezone to UTC for instantaneous data
        if not df.empty:
            df = df.tz_convert('UTC')
    
    if df.empty:
        print(f'No data available for station {site_code} in the specified time period')
        return None
    
    # Find discharge data column
    discharge_cols = [col for col in df.columns if '00060' in col and 'cd' not in col]
    if not discharge_cols:
        print(f"Error: No discharge column found for station {site_code}")
        return None
    
    discharge_col = discharge_cols[0]
    print(f"Found discharge column: {discharge_col}")
    
    # Create result DataFrame
    result_df = pd.DataFrame()
    result_df['datetime'] = df.index.copy()
    
    # Convert discharge from cfs to m³/s (conversion factor: 0.0283168)
    discharge_values = df[discharge_col].values * 0.0283168
    result_df['discharge_m3s'] = discharge_values
    
    # Add original values in cfs for reference
    result_df['discharge_cfs'] = df[discharge_col].values
    
    # Handle NaN values if they exist
    if result_df['discharge_m3s'].isna().any():
        print("Warning: NaN values detected in discharge data")
        nan_count = result_df['discharge_m3s'].isna().sum()
        total_count = len(result_df)
        print(f"NaN values: {nan_count}/{total_count} ({nan_count/total_count*100:.1f}%)")
    
    # Resample data if needed for hourly data
    if time_step == '1h' and service == 'iv':
        print("Resampling to hourly data...")
        result_df.set_index('datetime', inplace=True)
        result_df = result_df.resample('1H').mean()
        result_df.reset_index(inplace=True)
    
    # Create filename based on time step
    time_step_suffix = time_step.replace('min', 'm')  # 15min -> 15m
    output_file = os.path.join(output_dir, f'USGS_{site_code}_{time_step_suffix}_UTC.csv')
    
    # Save as CSV file
    result_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f'Successfully downloaded {len(result_df)} records')
    print(f'Data saved to: {output_file}')
    
    # Print data summary
    print("\nData Summary:")
    print(f"Records: {len(result_df)}")
    print(f"Date range: {result_df['datetime'].min()} to {result_df['datetime'].max()}")
    if not result_df['discharge_m3s'].isna().all():
        print(f"Discharge range: {result_df['discharge_m3s'].min():.3f} - {result_df['discharge_m3s'].max():.3f} m³/s")
        print(f"Mean discharge: {result_df['discharge_m3s'].mean():.3f} m³/s")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(
        description='Download USGS gauge discharge data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download daily data
  python usgs_gauge_download.py --site_num 08165300 --time_start 2025070400 --time_end 2025070523 --time_step 1d
  
  # Download hourly data
  python usgs_gauge_download.py --site_num 08165300 --time_start 2025070400 --time_end 2025070523 --time_step 1h
  
  # Download 15-minute data (default, highest resolution)
  python usgs_gauge_download.py --site_num 08165300 --time_start 2025070400 --time_end 2025070523
  
  # Custom output directory
  python usgs_gauge_download.py --site_num 08165300 --time_start 2025070400 --time_end 2025070523 --output /path/to/data
        """
    )
    
    parser.add_argument('--site_num', 
                        required=True, 
                        help='USGS station ID (e.g., 08165300)')
    
    parser.add_argument('--time_start', 
                        required=True, 
                        help='Start time in YYYYMMDDHH format (e.g., 2025070400)')
    
    parser.add_argument('--time_end', 
                        required=True, 
                        help='End time in YYYYMMDDHH format (e.g., 2025070523)')
    
    parser.add_argument('--time_step', 
                        default='15min',
                        choices=['1d', '1h', '15min'],
                        help='Time resolution: 1d (daily), 1h (hourly), 15min (15-minute, default)')
    
    parser.add_argument('--output', 
                        default=None,
                        help='Output directory (default: USGS_gauge/<site_num>/)')
    
    args = parser.parse_args()
    
    # Parse time strings
    start_date = parse_time_string(args.time_start)
    end_date = parse_time_string(args.time_end)
    
    # Validate time range
    if start_date >= end_date:
        print("Error: Start time must be before end time")
        sys.exit(1)
    
    # Set output directory
    if args.output is None:
        output_dir = os.path.join('USGS_gauge', args.site_num)
    else:
        output_dir = args.output
    
    # Download data
    try:
        df = download_usgs_data(
            site_code=args.site_num,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            time_step=args.time_step
        )
        
        if df is not None:
            print("\n✓ Download completed successfully!")
        else:
            print("\n✗ Download failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during download: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()