#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    [*] Description : Polls Analyzer
    [*] Author      : Dimitrios Georgiou (dgeorgiou3@gmail.com)
    [*] Date        :
    [*] Links       :
"""
# -*-*-*-*-*-*-*-*-*-*-* #
#     Basic Modules      #
# -*-*-*-*-*-*-*-*-*-*-* #
import os, json, time, glob,argparse, re
import numpy     as np
import functools as ft
import pandas    as pd
import datetime  as dt
import streamlit as st
from urllib.parse    import unquote
from bs4             import BeautifulSoup
from tqdm            import tqdm
from typing          import Dict, Any, List
from IPython.display import display
from concurrent.futures import ThreadPoolExecutor

# -*-*-*-*-*-*-*-*-*-*-* #
#     Project Modules    #
# -*-*-*-*-*-*-*-*-*-*-* #
# framework
from lib.framework.markI import *

# handlers
from lib.handlers.data_handling   import DataLoader
from lib.handlers.survey_handling import SurveyHandler

# modules
from lib.modules.API_google import (
    GoogleAPI, GoogleSheetsAPI, GoogleEmailAPI, GoogleDocsAPI, GoogleDriveAPI
)

# helpers
import lib.helpers.utils as utils


class Controller():
    def __init__(self, mk1 : MkI) :
        self.mk1  = mk1
        self.args = self.parsing()

        self.__null_values = [
            None, np.nan, "", "#N/A", "null", "nan", "NaN"
        ]


    def parsing(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--days_diff",
            "-dd",
            type    = int,
            default = 0,
            help    = "Options = {..,-2,-1,0, 1, 2,...}"
        )

        parser.add_argument(
            "--questionnaire_name",
            "-qn",
            type    = str,
            default = "PHQ-9",
            help    = "Options ~ {PHQ-9,GAD-7, ...}"
        )
        return parser.parse_args()



    def run_initialization(self):
        # Initializing Modules
        self.google_api = GoogleAPI(
            mk1 = self.mk1
        )
        self.google_sheets_api = GoogleSheetsAPI(
            mk1        = self.mk1,
            google_api = self.google_api
        )
        self.google_email_api  = GoogleEmailAPI(
            mk1        = self.mk1,
            google_api = self.google_api
        )
        self.google_drive_api  = GoogleDriveAPI(
            mk1        = self.mk1,
            google_api = self.google_api
        )
        self.google_docs_api = GoogleDocsAPI(
            mk1        = self.mk1,
            google_api = self.google_api
        )
        
        # Initializing Handlers
        self.data_loader = DataLoader(
            mk1               = self.mk1,
            google_sheets_api = self.google_sheets_api
        )

        self.survey_handler = SurveyHandler(
            mk1         = self.mk1,
            data_loader = self.data_loader
        )
    

    def _refresh_session(self):
        self.run_initialization()


    def run_get_survey_responses(self):
        tqdm.pandas()

        ## _______________ *** Configuration (attributes) *** _______________ #
        # Google Sheets
        # sheets_reporter_id = st.secrets["google_sheets"]["reporter_id"]
        # sheets_reporter_tab_survey_results = st.secrets["google_sheets"]["reporter_tab_survey_results"]
        sheets_reporter_id = self.mk1.config.get("google_sheets","reporter_id")
        sheets_reporter_tab_survey_results = self.mk1.config.get("google_sheets","reporter_tab_survey_results")

        # App Static
        img_path_background = self.mk1.config.get("app_static","img_path_background")
        fn_questionnaires   = self.mk1.config.get("app_static","fn_questionnaires")
        
        # Attributes
        today = datetime.now()

        # Args
        questionnaire_name = self.args.questionnaire_name

        ## _____________________________________________________________________________________________________________________ ##
        ## 1. Get questionnaire
        questionnaire = self.survey_handler.get_questionnaire(
            fn_questionnaires  = fn_questionnaires,
            questionnaire_name = questionnaire_name
        )

        ## _____________________________________________________________________________________________________________________ ##
        ## 2. Set background
        self.survey_handler.set_background(
            image_path = img_path_background
        )
        
        ## _____________________________________________________________________________________________________________________ ##
        ## 3. (DataFrame Operations) Initialize logs dictionary
        logs = {
            "exec_date"      : today.strftime('%Y-%m-%d'),
            #"start_datetime" : (today - dt.timedelta(hours = hours)).strftime('%Y-%m-%d %H:%M:%S'),
            #"end_datetime"   : today.strftime('%Y-%m-%d %H:%M:%S'),
            #"period"         : hours
        }

        ## 4. Get Survey response & logs
        response = self.survey_handler.get_survey_result(
            questionnaire = questionnaire
        )

        ## _____________________________________________________________________________________________________________________ ##
        ## 5. Log Survey responses to google sheets
        self.survey_handler.log_survey_result(
            sheets_reporter_id = sheets_reporter_id,
            sheets_reporter_tab_survey_results = sheets_reporter_tab_survey_results
        )
        

        ## _____________________________________________________________________________________________________________________ ##
        ## 3. (Google Sheets API) Appends `logs` at Google Sheets
        # self.data_loader.append_data_to_google_sheets(
        #     df                     = logs,
        #     spreadsheet_id         = sheets_reporter_id,
        #     spreadsheet_range_name = sheets_reporter_tab_survey_results,
        # )



    ## -------------------------------------------------------------------------------------------------------------------------------------------- ##
    def run(self):
        # initialize services
        self.run_initialization()
        self.mk1.logging.logger.info(f"(Controller.run) All services initilalized")

        self.run_get_survey_responses()
        
        

if __name__ == '__main__':
    Controller(
        MkI.get_instance(_logging = True)
    ).run()
