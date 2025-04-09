#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    [*] Description : Polls Analyzer
    [*] Author      : Dimitrios Georgiou (dgeorgiou3@gmail.com)
    [*] Date        : JAN2025
    [*] Links       :
"""
# -*-*-*-*-*-*-*-*-*-*-* #
#     Basic Modules      #
# -*-*-*-*-*-*-*-*-*-*-* #
import os
import argparse
import functools as ft
import pandas    as pd
import datetime  as dt
import streamlit as st
from tqdm            import tqdm
from typing          import Dict, Any, List
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pingouin as pg



# -*-*-*-*-*-*-*-*-*-*-* #
#     Project Modules    #
# -*-*-*-*-*-*-*-*-*-*-* #
# framework
from lib.framework.markI import *

# handlers
from lib.handlers.data_handling        import DataLoader
from lib.handlers.data_preprocessing   import DataPreprocessor
from lib.handlers.survey_handling      import SurveyHandler
from lib.handlers.statistical_analysis import *
from lib.handlers.report_handling      import ReportHandler

# modules
from lib.modules.API_google import (
    GoogleAPI, GoogleSheetsAPI, GoogleEmailAPI, GoogleDocsAPI, GoogleDriveAPI
)
from lib.modules.API_openai import OpenaiAPI
from lib.modules.API_dropbox import DropboxAPI

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

        ## ______________________________ General ______________________________ ##
        parser.add_argument(
            "--operation",
            "-o",
            type    = str,
            default = "survey",
            help    = "Options = {statistical_report_generation,statistical_analysis,survey}",
        )
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

        ## ______________________________ Statistical Tests ______________________________ ##
        parser.add_argument( "--significance_threshold", type = float, default = 0.05, help = 'In the range [0,1]')

        parser.add_argument( "--chi_square_dof_min", type = int, default = 9.0, help = '1 or more')
        parser.add_argument( "--chi_square_cramer_v_min", type = float, default = 0.1, help = 'In the range [0,1]')
        parser.add_argument( "--chi_square_power_min", type = float, default = 0.8, help = 'In the range [0,1]')

        parser.add_argument( "--anova_power_min", type = float, default = 0.5, help = 'In the range [0,1]')
        parser.add_argument( "--anova_cohens_d_min", type = float, default = 0.3, help = 'In the range [0,1]')
        parser.add_argument( "--anova_epsilon_squared_min", type = float, default = 0.03, help = 'In the range [0,1]')
        parser.add_argument( "--anova_eta_squared_min", type = float, default = 0.03, help = 'In the range [0,1]')
        parser.add_argument( "--anova_cles_diff_min", type = float, default = 0.1, help = 'In the range [0,1]')

        parser.add_argument( "--nonparam_power_min", type = float, default = 0.5, help = 'In the range [0,1]')
        parser.add_argument( "--nonparam_cles_diff_min", type = float, default = 0.05, help = 'In the range [0,1]')
        parser.add_argument( "--nonparam_epsilon_squared_min", type = float, default = 0.02, help = 'In the range [0,1]')

        parser.add_argument( "--pearson_corr_min", type = float, default = 0.55, help = 'In the range [0,1]')
        parser.add_argument( "--pearson_power_min", type = float, default = 0.6, help = 'In the range [0,1]')

        parser.add_argument( "--spearman_corr_min", type = float, default = 0.55, help = 'In the range [0,1]')
        parser.add_argument( "--spearman_power_min", type = float, default = 0.6, help = 'In the range [0,1]')

        
        return parser.parse_args()
    



    def run_initialization(self, use_openai: bool = True):
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

        if use_openai : 
            self.openai_api = OpenaiAPI(
                mk1 = self.mk1
            )
        
        # Initializing Handlers
        self.data_loader = DataLoader(
            mk1               = self.mk1,
            google_sheets_api = self.google_sheets_api
        )

        self.data_preprocessor = DataPreprocessor(
            mk1 = self.mk1,
        )

        self.survey_handler = SurveyHandler(
            mk1         = self.mk1,
            data_loader = self.data_loader
        )
        
        self.chi_square_analyzer     = ChiSquareAnalyzer()
        self.fishers_exact_analyzer  = FishersExactAnalyzer()
        self.anova_analyzer          = ANOVAAnalyzer()
        self.descriptive_post_statistics_analyzer =  DescriptivePostStatisticsAnalyzer()
        self.statistical_tester = StatisticalTests(
            significance_threshold = self.args.significance_threshold
        )

        self.report_handler = ReportHandler(
            mk1              = self.mk1,
            google_email_api = self.google_email_api,
            openai_api       = self.openai_api
        )
    

    def _refresh_session(self):
        self.run_initialization()

    def _refresh_tokens(self):
        ## _______________ *** Configuration (objects) *** _______________ #
        self.dropbox_api = DropboxAPI(
            mk1 = self.mk1
        )

        ## _______________ *** Configuration (attributes) *** _______________ #
        google_oauth_accessed_dbx_path   = self.mk1.config.get("dropbox", "google_oauth_accessed_dbx_path")
        google_oauth_local_path          = self.mk1.config.get("api_google", "token_file_path")
        google_oauth_accessed_local_path = f"{google_oauth_local_path.rsplit('.', 1)[0]}_accessed.json"

        ## _____________________________________________________________________________________________________________________ ##
        self.dropbox_api.download_file(
            dropbox_path = google_oauth_accessed_dbx_path,
            local_path   = google_oauth_accessed_local_path
        ) 


    def run_get_survey_responses(self):
        tqdm.pandas()

        ## _______________ *** Configuration (attributes) *** _______________ #
        # Google Sheets
        sheets_reporter_id = self.mk1.config.get("google_sheets","reporter_id")
        sheets_reporter_tab_survey_results = self.mk1.config.get("google_sheets","reporter_tab_survey_results")

        # App Static
        img_path_backgrounds = ast.literal_eval(self.mk1.config.get("app_static","img_path_backgrounds"))
        fn_questionnaires    = self.mk1.config.get("app_static","fn_questionnaires")
        
        # Args
        questionnaire_name   = self.args.questionnaire_name
        img_path_background = img_path_backgrounds[
            questionnaire_name
        ]
        sheets_reporter_tab_survey_results = sheets_reporter_tab_survey_results.format(
            questionnaire_name = questionnaire_name
        )


        ## _____________________________________________________________________________________________________________________ ##
        ## 1. Get questionnaire
        questionnaire, _ = self.survey_handler.get_questionnaire(
            fn_questionnaires  = fn_questionnaires,
            questionnaire_name = questionnaire_name
        )

        ## _____________________________________________________________________________________________________________________ ##
        ## 2. Set background
        self.survey_handler.set_background(
            image_path = img_path_background,
            opacity    = 0.2
        )
        
        ## _____________________________________________________________________________________________________________________ ##
        ## 3. Get Survey response & logs
        self.survey_handler.get_survey_result(
            questionnaire = questionnaire
        )

        ## _____________________________________________________________________________________________________________________ ##
        ## 4. Log Survey responses to google sheets
        self.survey_handler.log_survey_result(
            sheets_reporter_id                 = sheets_reporter_id,
            sheets_reporter_tab_survey_results = sheets_reporter_tab_survey_results
        )



    def generate_folders(self, dir_static : str, questionnaire_name:str ): 
        if not os.path.exists(f"{dir_static}/{questionnaire_name}"):
            os.makedirs(f"{dir_static}/{questionnaire_name}")

        if not os.path.exists(f"{dir_static}/{questionnaire_name}/descriptive"):
            os.makedirs(f"{dir_static}/{questionnaire_name}/descriptive")

        if not os.path.exists(f"{dir_static}/{questionnaire_name}/categorical-categorical"):
            os.makedirs(f"{dir_static}/{questionnaire_name}/categorical-categorical")

        if not os.path.exists(f"{dir_static}/{questionnaire_name}/continuous-continuous"):
            os.makedirs(f"{dir_static}/{questionnaire_name}/continuous-continuous")

        if not os.path.exists(f"{dir_static}/{questionnaire_name}/categorical-continuous"):
            os.makedirs(f"{dir_static}/{questionnaire_name}/categorical-continuous")

    
    ## ____________________________________________________________________________________________________________________________________________________________________________________ ##
    def run_statistical_analysis(self) :
        tqdm.pandas()

        ## _______________ *** Configuration (attributes) *** _______________ #
        # Args
        questionnaire_name  = self.args.questionnaire_name

        # app
        dir_static = self.mk1.config.get("app","dir_static")

        # app static
        fn_questionnaires = self.mk1.config.get("app_static","fn_questionnaires")

        # google sheets
        sheets_reporter_id = self.mk1.config.get("google_sheets","reporter_id")
        sheets_reporter_tab_survey_results = self.mk1.config.get("google_sheets","reporter_tab_survey_results")
        sheets_reporter_tab_survey_results = sheets_reporter_tab_survey_results.format(
            questionnaire_name = questionnaire_name
        ) + "!A2:N"

        ## ____________________________________________________________ #
        self.generate_folders(
            dir_static = dir_static,
            questionnaire_name = questionnaire_name
        )

        ## ____________________________________________________________ #
        """ 1. Data Loading """
        _, questions_mapping = self.survey_handler.get_questionnaire(
            fn_questionnaires  = fn_questionnaires,
            questionnaire_name = questionnaire_name
        )

        data = self.data_loader.load_data_from_google_sheets_tab(
            spreadsheet_id         = sheets_reporter_id,
            spreadsheet_range_name = sheets_reporter_tab_survey_results,
        )
        
        ## ____________________________________________________________ #
        """
            2. Data Preprocessing
                2.1 Preprocess data types
                2.2 Proprocess data overall, pivot on questions!
        """
        data = self.data_preprocessor.preprocess_dtypes(
            data = data
        )

        data = self.data_preprocessor.preprocess_city(
            data = data
        )

        # data = self.data_preprocessor.get_country(
        #     data = data
        # )

        data = self.data_preprocessor.pivot_on_questions(
            data = data
        )
          
    
        ## ____________________________________________________________ #
        """ 3. Analytics
                3.1 Descriptive Statistics
                3.2 Statistical Tests by Variable Type
                    - Categorical-Categorical: Chi-square/Fisher's Exact
                    - Categorical-Continuous: ANOVA/t-test and nonparametric alternatives
                    - Continuous-Continuous: Correlation tests
        """
       # 3.1 (Descriptive Statistics) 
        self.descriptive_post_statistics_analyzer.create_summary_visualizations(
            data_df   = data,
            directory = f"{dir_static}/{questionnaire_name}/descriptive"
        )
        self.descriptive_post_statistics_analyzer.create_demographic_visualizations(
            data_df   = data,
            directory = f"{dir_static}/{questionnaire_name}/descriptive"
        )

        self.descriptive_post_statistics_analyzer.create_distribution_visualizations(
            data_df   = data,
            directory = f"{dir_static}/{questionnaire_name}/descriptive"
        )

        self.descriptive_post_statistics_analyzer.create_advanced_visualizations(
            data_df   = data,
            directory = f"{dir_static}/{questionnaire_name}/descriptive"
        )

        # self.descriptive_post_statistics_analyzer.create_geographic_heatmap(
        #     data_df   = data,
        #     directory = f"{dir_static}/{questionnaire_name}/descriptive"
        # )
          
    
        ## ____________________________________________________________ #
        # 3.2 Statistical Tests
        # Auto-detect variable pairs
        pairs = self.statistical_tester.get_pairs_of_variables(data)


        ## ____________________________________________________________ #
        # 3.2.1 Categorical-Categorical: Chi-square tests
        chi2_results = self.statistical_tester.chi2_test_wrapper(
            data         = data,
            pairs        = pairs["categorical_categorical"],
            directory    = f"{dir_static}/{questionnaire_name}/categorical-categorical",
            dof_min      = self.args.chi_square_dof_min,
            cramer_v_min = self.args.chi_square_cramer_v_min,
            power_min    = self.args.chi_square_power_min
        )
        
        
        ## ____________________________________________________________ #
        # 3.2.2 Categorical-Continuous: Parametric tests (ANOVA/t-tests)
        anova_results = self.statistical_tester.anova_test_wrapper(
            data      = data,
            pairs     = pairs["categorical_continuous"],
            directory = f"{dir_static}/{questionnaire_name}/categorical-continuous",
            power_min           = self.args.anova_power_min,
            cohens_d_min        = self.args.anova_cohens_d_min,
            epsilon_squared_min = self.args.anova_epsilon_squared_min,
            eta_squared_min     = self.args.anova_eta_squared_min,
            cles_diff_min       = self.args.anova_cles_diff_min
        )
        
        # 3.2.3 Categorical-Continuous: Non-parametric tests (Mann-Whitney/Kruskal-Wallis)
        nonparam_group_results = self.statistical_tester.nonparametric_group_test_wrapper(
            data                = data,
            pairs               = pairs["categorical_continuous"],
            directory           = f"{dir_static}/{questionnaire_name}/categorical-continuous",
            epsilon_squared_min = self.args.nonparam_epsilon_squared_min,
            cles_diff_min       = self.args.nonparam_cles_diff_min,
            power_min           = self.args.nonparam_power_min
        )

       
        ## ____________________________________________________________ #
        # 3.2.4. Continuous-Continuous: Correlation tests
        # Try both Pearson (parametric) and Spearman (non-parametric) correlations
        pearson_results = self.statistical_tester.correlation_test_wrapper(
            data      = data,
            pairs     = pairs["continuous_continuous"],
            method    = 'pearson',
            directory = f"{dir_static}/{questionnaire_name}/continuous-continuous",
            corr_min  = self.args.pearson_corr_min,
            power_min = self.args.pearson_power_min
        )

        spearman_results = self.statistical_tester.correlation_test_wrapper(
            data      = data,
            pairs     = pairs["continuous_continuous"],
            method    = 'spearman',
            directory = f"{dir_static}/{questionnaire_name}/continuous-continuous",
            corr_min  = self.args.spearman_corr_min,
            power_min = self.args.spearman_power_min
        )

        ## ____________________________________________________________ #
        # 3.3 Save ALL results to CSV
        chi2_results.to_csv(
            path_or_buf = f"{dir_static}/{questionnaire_name}/categorical-categorical/chi2_results.csv", 
            index       = False
        )

        anova_results.to_csv(
            path_or_buf = f"{dir_static}/{questionnaire_name}/categorical-continuous/anova_results.csv", 
            index       = False
        )
        nonparam_group_results.to_csv(
            path_or_buf = f"{dir_static}/{questionnaire_name}/categorical-continuous/nonparametric_results.csv", 
            index       = False 
        )
        pearson_results.to_csv(
            path_or_buf = f"{dir_static}/{questionnaire_name}/continuous-continuous/pearson_results.csv", 
            index       = False
        )
        spearman_results.to_csv(
            path_or_buf = f"{dir_static}/{questionnaire_name}/continuous-continuous/spearman_results.csv", 
            index       = False
        )

        ## ____________________________________________________________ #
        # 3.4 Generate a summary of significant findings across all tests
        filters = self.statistical_tester.get_all_filters(
            args = self.args
        )

        self.statistical_tester.generate_overall_summary(
            dir_static             = dir_static,
            questionnaire_name     = questionnaire_name,
            filters                = filters,
            chi2_results           = chi2_results,
            anova_results          = anova_results,
            nonparam_group_results = nonparam_group_results,
            pearson_results        = pearson_results,
            spearman_results       = spearman_results,
        )


    

    def run_statistical_report_generation(self) :
        tqdm.pandas()

        ## _______________ *** Configuration (attributes) *** _______________ #
        # Args
        questionnaire_name  = self.args.questionnaire_name

        # app
        dir_static = self.mk1.config.get("app","dir_static")

        # app static
        fn_questionnaires = self.mk1.config.get("app_static","fn_questionnaires")
        fn_technical_summary     = f"./{dir_static}/{questionnaire_name}/significant_findings_summary.txt"
        fn_non_technical_summary = f"./{dir_static}/{questionnaire_name}/descriptive/key_findings_summary.txt"
        fn_report_html           = f"./{dir_static}/{questionnaire_name}/final_report_{questionnaire_name}.html"
        fn_report_pdf            = f"./{dir_static}/{questionnaire_name}/final_report_{questionnaire_name}.pdf"
        

        # google sheets
        sheets_reporter_id = self.mk1.config.get("google_sheets","reporter_id")
        sheets_reporter_tab_survey_results = self.mk1.config.get("google_sheets","reporter_tab_survey_results")
        sheets_reporter_tab_survey_results = sheets_reporter_tab_survey_results.format(
            questionnaire_name = questionnaire_name
        ) + "!A2:N"

        # google_email
        cc = self.mk1.config.get("google_email","cc")


        ## ____________________________________________________________ #
        """ 1. Data Loading 
            - Questionnaire from file
            - Data from survey results
        """
        questionnaire, questions_mapping = self.survey_handler.get_questionnaire(
            fn_questionnaires  = fn_questionnaires,
            questionnaire_name = questionnaire_name
        )

        data = self.data_loader.load_data_from_google_sheets_tab(
            spreadsheet_id         = sheets_reporter_id,
            spreadsheet_range_name = sheets_reporter_tab_survey_results,
        )


        ## ____________________________________________________________ #
        """ 2. Analytics from images
                2.1 Get all the paths
                2.2 (OpenAI) Generate analytics 
                2.3 Generate the final .html report
                2.4 Save the .html report
                2.4 Save the .html report to .pdf report

            Testing Purposes
        """
        # Get all the image paths
        image_paths_dict = self.statistical_tester.get_all_visualization_paths(
            dir_static          = dir_static,
            questionnaire_name  = questionnaire_name
        )

        image_paths = [
            img_path for category in image_paths_dict.values() for img_path in category if '.DS_Store' not in img_path
        ]
        
        if not os.path.exists(fn_report_pdf):
            # Generate analytics
            analysis_results = self.openai_api.analyze_statistical_images(
                image_paths       = image_paths,
                questions_mapping = questions_mapping
            )

            # Generate the .html report
            report_html_content = self.report_handler.generate_statistical_report(
                analysis_results   = analysis_results,
                report_title       = f"Statistical Analysis Report for {questionnaire['title']}",
                questionnaire_name = questionnaire_name,
                questions_mapping  = questions_mapping,
                technical_summary  = fn_technical_summary
            )
        
            # Save the .html report
            self.report_handler.save_report_to_file(
                html_content = report_html_content, 
                output_path  = fn_report_html
            ) 

            # Convert .html to .pdf
            self.report_handler.html_to_pdf(
                html_path = fn_report_html
            )

        # Send .pdf (converted from .html) report by email to all participants
        self.report_handler.send_report_by_email(
            report_path              = fn_report_pdf,
            df                       = data,
            cc                       = cc,
            subject                  = f"📊 Your Analysis Report for {questionnaire_name}",
            fn_non_technical_summary = fn_non_technical_summary
        )

        
          




    ## ____________________________________________________________________________________________________________________________________________________________________________________ ##
    def run(self):
        # args
        operation = self.args.operation

        # initialize services
        self._refresh_tokens()
        

        # actions
        if operation == "statistical_analysis" :
            self.run_initialization()
            self.run_statistical_analysis()

        elif operation == "statistical_report_generation":
            self.run_initialization()
            self.run_statistical_report_generation()

        elif operation == "survey" :
            self.run_initialization(use_openai = False)
            self.run_get_survey_responses()
        
        

if __name__ == '__main__':
    Controller(
        MkI.get_instance(_logging = True)
    ).run()
