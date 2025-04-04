# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    [*] Description : Data Handling I/O operations from .txt, google sheets, etc
    [*] Author      : dgeorgiou3@gmail.com
    [*] Date        : Jan, 2024
    [*] Links       :
"""

# -*-*-*-*-*-*-*-*-*-*-* #
#     Basic Modules      #
# -*-*-*-*-*-*-*-*-*-*-* #
import os, json, ast, itertools
import numpy    as np
import pandas   as pd
import datetime as dt
from urllib.parse           import unquote
from unidecode              import unidecode
from retry                  import retry
from tqdm                   import tqdm
from dateutil.relativedelta import relativedelta
from datetime               import datetime
from IPython.display        import display
from typing                 import Dict, Any, List

# -*-*-*-*-*-*-*-*-*-*-* #
#     Project Modules    #
# -*-*-*-*-*-*-*-*-*-*-* #
class DataPreprocessor():
    def __init__(
            self,
            mk1,
        ) :
        ## System Design
        self.mk1 = mk1


    

    def preprocess_dtypes(self, data : pd.DataFrame) -> pd.DataFrame:
        data['Question (id)'] = pd.to_numeric(data['Question (id)'], errors='coerce')
        data['Answer']        = pd.to_numeric(data['Answer'], errors='coerce')
        return data

    def preprocess_city(self, data : pd.DataFrame) -> pd.DataFrame:
        data['City'] = data['City'].apply(lambda x: unidecode(x.lower().strip().split(',')[0]))
        return data

    def _get_country_row(self, city : str) -> str:
        # Initialize geolocator
        geolocator = Nominatim(user_agent="city_to_country")

        location = geolocator.geocode(city)
        if location:
            return location.address.split(',')[-1].strip()  # Extract the last part as the country
        else:
            return None  # In case no location is found

    def get_country(self, data : pd.DataFrame) -> pd.DataFrame:
        data['Country'] = data['City'].progress_apply(lambda x: self._get_country_row(x))
        return data



    def pivot_on_questions(self, data : pd.DataFrame) -> pd.DataFrame:
        # Identify demographic columns (they stay the same in each group of 10 rows)
        person_columns = ['Timestamp', 'Start Time', 'End Time', 'Age Group', 'Gender', 
                        'Educational Level', 'Employment Status', 'Living Situation', 
                        'Physical Activity Level', 'City', 'Email']

        # Get total number of rows
        num_rows = len(data)

        # Create a list to store merged rows
        merged_data = []

        # Iterate through the DataFrame in chunks of 10 rows
        for i in range(0, num_rows, 10): 
            chunk = data.iloc[i:i+10]  # Get the next 10 rows

            # Take demographic details from the first row in the chunk
            person_info = chunk.iloc[0][person_columns].to_dict()

            # Pivot the chunk to get question columns
            pivoted = chunk.pivot(index='Email', columns='Question (id)', values='Answer')

            # Convert question numbers to Q1, Q2, ..., Q10
            pivoted.columns = [f"Q{int(col)}" for col in pivoted.columns]

            # Drop index to make it a single row
            pivoted = pivoted.reset_index(drop=True)

            # Merge the demographic data with the answers
            merged_row = {**person_info, **pivoted.iloc[0].to_dict()}

            # Store the merged row
            merged_data.append(merged_row)

        # Convert merged data back to DataFrame
        data_cleaned = pd.DataFrame(merged_data)
        return data_cleaned


  