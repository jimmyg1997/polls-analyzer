# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    [*] Description : Py3 class for MarkI system design for all frameworks
    [*] Author      : Dimitrios Georgiou (dgeorgiou3@gmail.com)
    [*] Date        : Feb, 2024
    [*] Links       :
"""

# -*-*-*-*-*-*-*-*-*-*-* #
#     Basic Modules      #
# -*-*-*-*-*-*-*-*-*-*-* #
import os, json, base64, re
import numpy    as np
import pandas   as pd
import datetime as dt
import streamlit as st
import streamlit.components.v1 as components

from retry                  import retry
from tqdm                   import tqdm
from dateutil.relativedelta import relativedelta
from datetime               import datetime
from IPython.display        import display
from typing                 import Dict, Any, List

# -*-*-*-*-*-*-*-*-*-*-* #
#     Project Modules    #
# -*-*-*-*-*-*-*-*-*-*-* #

from lib.helpers.utils import *

class SurveyHandler():
    def __init__(
            self,
            mk1,
            data_loader
        ) :
        ## System Design
        self.mk1 = mk1

        ## ______________ ** APIs & Handlers ** ______________ ##
        self.data_loader = data_loader

        ## ______________ ** Attributes ** ______________ ##
        
   
    def get_questionnaire(
            self, 
            fn_questionnaires  : str,
            questionnaire_name : str 
        ) -> Dict[str,str]: 

        try : 
            with open(fn_questionnaires, "r", encoding="utf-8") as file:
                questionnaires = json.load(file) 
            questionnaire = questionnaires[questionnaire_name]
            return questionnaire
        except Exception as e :
            raise e


    def set_background(
            self, 
            image_path : str, 
            opacity    : float = 0.3
        ):
        """
        Sets a background image in a Streamlit app with reduced opacity.

        Parameters:
        - image_url (str): Direct URL to the background image (GitHub raw link).
        - opacity (float): Opacity level (0.0 to 1.0, where 1 is fully visible and 0 is fully transparent).
        """
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(rgba(255,255,255, {1-opacity}), rgba(255,255,255, {1-opacity})), 
                            url("{image_path}") no-repeat center center fixed;
                background-size: cover;
            }}
        
            
            /* _____________ General _____________ */
            /* Ensures text is always visible */
            
            html, body, [class*="st-"] {{
                color: black !important;  /* Default color for light mode */
            }}

            /* Dark mode override */
            @media (prefers-color-scheme: dark) {{
                html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
                    color: black !important; 
                }}

                /* Ensure all text stays black */
                * {{
                    color: black !important;
                }}
            }}

            /* Change the top bar background color */
            header[data-testid="stHeader"] {{
                background-color: #1E1E1E !important;  /* Dark Gray */
            }}

            /* Change the text color of the top bar buttons (Deploy, Settings, etc.) */
            header[data-testid="stHeader"] * {{
                color: white !important;
            }}

            ul[role="listbox"] li {{
                color: white !important;
                background-color: #FFFFFF;
                font-weight: bold;      
                font-size: 16px;
            }}

            div[role="dialog"]  {{
                color: white !important;
                background-color: #FFFFFF;
                font-weight: bold;      
                font-size: 16px;
            }}
            
            /* Fix text for dark mode */
            @media (prefers-color-scheme: dark) {{
                html, body, [class*="st-"] {{
                    color: white !important;
                }}
            }}

            /* _____________ 1. stSelectbox _____________ */
            /* Fix font color for selectbox */
            ul[data-testid="stSelectboxVirtualDropdown"] li {{
                color: white !important;
                background-color: #FFFFFF;
                font-weight: bold;      
                font-size: 16px;
            }}
           
            .stSelectbox div[data-baseweb="select"] > div:first-child {{
                background-color: #FFFFFF;
                border-color: #2d408d;
            }}

            /* _____________ 2. stTextInput _____________ */
            div[data-testid="stTextInput"] {{
                color: white !important;
                border-color: #2d408d;
            }}

            .stTextInput > div > div > input {{
                background-color: #FFFFFF;
                border-color: #2d408d;
            }}

            div[data-testid="stTextInput"] > div:focus-within {{
                box-shadow: 0px 0px 5px 2px #2d408d !important;
                border-color: #2d408d !important;
            }}

            /* _____________ 3. checkBox _____________ */
            div[data-baseweb="checkbox"] input {{
                color: white !important;
                border-color: #2d408d;
            }}
            div[data-baseweb="checkbox"] > div > div{{
                color: white !important;
                border-color: #2d408d;
            }}

            /* _____________ 4. stButton _____________ */
            /* Fix font color for buttons */
            .stButton>button {{
                color: white !important;
                background-color: #FFFFFF;
                border-color: #2d408d;
                border-radius: 8px;
                font-weight: bold;
            }}

            /* _____________ 5. stSlider _____________ */    
            /* Ensure slider text is readable */
            .stSlider label {{
                color: white !important;
            }}

            /* _____________ 5. stFileUploader _____________ */

            /* Fix file uploader */
            .stFileUploader label {{
                color: white !important;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )

    def get_headers_settings(self):
        header_settings = """
            <style>
            /* Responsive title */
            h1 {
                font-size: 2.2vw;  /* Scales with screen width */
                display: flex;
                align-items: center;
            }

            /* Mobile adjustments */
            @media (max-width: 768px) {
                h1 {
                    font-size: 4.5vw; /* Adjust for mobile */
                }
            }

            /* Image styling to match text height */
            h1 img {
                height: 1em; /* Matches text height */
                margin-left: 10px; /* Spacing between text and image */
            }
            </style>
            """
        return header_settings


    def get_survey_result(
            self,
            questionnaire  : Dict[str,str]
        ):
        

        ## ______________ ** Config ** ______________ ##
        header_settings = self.get_headers_settings()
        title           = questionnaire['title']
        purpose         = questionnaire['purpose']
        questions_dict  = {k: v for k, v in questionnaire.items() if k.startswith("question")}
        choices         = questionnaire['choices']
        help_text       = ", ".join(f"{i}: {choice}" for i, choice in choices.items())
        


        ## ___________________________________________________________________________ ##
        st.markdown(
            header_settings + f""" <h1> {title} </h1> <h3> {purpose} </h3>""",
            unsafe_allow_html=True
        )
        st.markdown(
            "*I, Dimitrios Georgiou, a 𝐃𝐚𝐭𝐚 𝐒𝐜𝐢𝐞𝐧𝐭𝐢𝐬𝐭, am conducting this survey to analyze trends and behavioral patterns based on demographic and lifestyle factors. The collected data will be used solely for statistical analysis, ensuring anonymity and secure processing. Insights from this study aim to enhance understanding of behavioral trends, support research, and contribute to informed decision-making. Participation is voluntary, and no personally identifiable information will be shared.*"
        )

        # Initialize session state for start time
        if "start_time" not in st.session_state:
            st.session_state.start_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create two columns
        left_column, right_column = st.columns([1, 4])  # Adjust the ratio of column width

        # Left column: Metadata
        with left_column:
            if "required_fields_filled" not in st.session_state:
                st.session_state.required_fields_filled = False

            #st.image("https://raw.githubusercontent.com/jimmyg1997/polls-analyzer/main/static/1.png", use_container_width=True)
            age_group               = st.selectbox("Age Group", ["", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84"])
            gender                  = st.selectbox("Gender", ["", "Male", "Female", "Non-binary", "Prefer not to say"])
            education_level         = st.selectbox("Education Level", ["", "High school", "Bachelor's", "Master's", "PhD" , "Other"])
            employment_status       = st.selectbox("Employment Status", ["", "Employed full-time", "Part-time", "Unemployed","Student", "Retired"])
            living_situation        = st.selectbox("Living Situation", ["", "Alone", "With partner", "With family", "Shared housing"])
            physical_activity_level = st.selectbox("Physical Activity Level", ["", "Sedentary", "Light activity", "Moderate activity", "High activity"])
            city                    = st.text_input("City", "")

            metadata = {
                "age_group"               : age_group,
                "gender"                  : gender,
                "education_level"         : education_level,
                "employment_status"       : employment_status,
                "living_situation"        : living_situation,
                "physical_activity_level" : physical_activity_level,
                "city"                    : city
            }
            # Later, before processing the data (e.g., on form submission), validate that no required field still has the placeholder.
            metadata_to_check = list(metadata.values())
            errors = []

            if any(field == "" for field in metadata_to_check) :
                errors.append("Please fill in all fields above")

            if errors:
                st.error("\n".join(errors))
            else:
                st.session_state.required_fields_filled = True
                st.success("All required fields are completed!")
                # Continue processing metadata...

            st.markdown(f"*Want to receive all the results once the poll ends? Enter your email below!*")
            email = st.text_input("Email", "")
            metadata["email"] = email
            

        # Right column: Questions
        with right_column:
            # Initialize session state to track responses
            if 'responses' not in st.session_state:
                st.session_state.responses = []
            
            # Start Questionnaire
            st.markdown(f"*{len(questions_dict)} questions, < 2 min to fill in*")
        
            # Display the choices above the slider in evenly spaced columns
            #st.markdown(f"**{help_text}**")

            for question_id, question in questions_dict.items():
                question_id = re.findall(r'\d+', question_id)[0]
                self.ask_q(
                    metadata    = metadata, 
                    question_id = question_id,
                    question    = question,
                    choices     = choices
                )


    def ask_q(
            self, 
            metadata    : Dict[str,str],
            question_id : int,
            question    : str,
            choices     : Dict[str,str],
        ):
        
        # Inject custom CSS for the slider thumb
        st.markdown(
            """
            <style>
            /* Target the container of the slider (BaseWeb slider) */
            [data-baseweb="slider"] {
                padding-left: 20px !important;
                padding-right: 20px !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        button_type = "slider"
        # Initialize the session state if not already initialized
        if f"{button_type}_value_{question_id}" not in st.session_state:
            st.session_state[f"{button_type}_value_{question_id}"] = 0  # Default value when the page first loads
            self.store_response(
                timestamp    = "",
                question_id  = question_id,
                question     = question,
                answer       = 0, 
                metadata     = metadata,
            )


        if button_type == "radio" : 
            answer = st.radio(
                f"**Q{question_id}** : {question}", 
                list(choices.keys()), 
                format_func=lambda x: choices[x],
                horizontal=True
            )

        elif button_type == "slider" : 
            answer = st.slider(
                f"**Q{question_id}** : {question}",
                min_value = 0, 
                max_value = len(choices) - 1,
                value     = 0,  # Default value (not preselected)
                step      = 1,
                format    = "%d",  # Display the slider value as an integer
            )            
            slider_container = f"""
                <style>
                    /* Hide the min/max values of the Streamlit slider */
                    div[data-testid="stSliderTickBar"] {{
                        display: none !important;
                    }}
                    div[data-testid="stSliderTickBarMin"] {{
                        display: none !important;
                    }}
                    div[data-testid="stSliderTickBarMax"] {{
                        display: none !important;
                    }}
                    div[data-testid="stSlider"] span {{
                        display: none !important;
                    }}
                    /* Positioning the labels */
                    .slider-container {{
                        position: relative;
                        width: 100%;
                    }}
                    .slider-labels {{
                        display: flex;
                        justify-content: space-between;
                        width: 100%;
                        position: absolute;
                        top: 5px;
                        font-size: 10px;
                        font-weight: bold;
                    }}

                    
                </style>
            """

            slider_container_labels = "<div class='slider-container'> <div class='slider-labels'>"
            for _, c in choices.items():
                slider_container_labels += f"<span>{c}</span>"

            slider_container_labels += "</div> </div>"

            st.markdown(
                slider_container + 
                slider_container_labels,
                unsafe_allow_html=True
            )


        # If the value has changed, capture the timestamp and store the response
        if answer != st.session_state[f"{button_type}_value_{question_id}"]:
            st.session_state[f"{button_type}_value_{question_id}"] = answer  # Update the session state with new value
            timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.store_response(
                timestamp    = timestamp,
                question_id  = question_id,
                question     = question,
                answer       = answer, 
                metadata     = metadata,
            )
            


    def store_response(
            self, 
            timestamp   : dt.datetime,
            question_id : int, 
            question    : str, 
            answer      : str, 
            metadata    : str, 
        ):

        logs_dict = {
            "Timestamp"     : timestamp,
            "Question (id)" : question_id,
            "Question"      : question,
            "Answer"        : answer,
        }

        response = merge_dicts(metadata, logs_dict)
        st.session_state.responses.append(response)



    def log_survey_result(
            self, 
            sheets_reporter_id                 : str, 
            sheets_reporter_tab_survey_results : str
        ):
        # Create two columns
        left_column, right_column = st.columns([1, 4])  # Adjust the ratio of column width
        required_fields_filled = st.session_state.required_fields_filled

        # Log results with timestamp
        with right_column :
            # Acceptance checkbox
            accept_terms = st.checkbox("Do you consent to the processing of your data for statistical analysis purposes?")

            if st.button("Submit Response", disabled=not required_fields_filled):
                if not accept_terms:
                    st.error("You must accept the terms and conditions to proceed.")

                else : 
                    if 'responses' in st.session_state:
                        # Converting responses to DataFrame
                        df = pd.DataFrame(st.session_state.responses)
                        print(df)

                        df["Question (id)"] = df["Question (id)"].astype(int)

                        df = df.sort_values('Timestamp')\
                            .drop_duplicates('Question', keep='last')\
                            .sort_values('Question (id)')

                        # add times
                        df['Start Time'] = st.session_state.start_time
                        df['End Time']   = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Save to CSV
                        st.success("Response submitted successfully!")

                        # Save to Google Sheets
                        self.data_loader.append_data_to_google_sheets(
                            df                     = df,
                            spreadsheet_id         = sheets_reporter_id,
                            spreadsheet_range_name = sheets_reporter_tab_survey_results,
                        )

                        return df
                    else:
                        st.warning("No responses available to create the dataframe.")
                        return None