# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    [*] Description : Py3 wrapper for OpenAI LM models
    [*] Author      : dgeorgiou3@gmail.com
    [*] Date        : MAR2025
    [*] Links       :
"""

# -*-*-*-*-*-*-*-*-*-*-* #
#     Basic Modules      #
# -*-*-*-*-*-*-*-*-*-*-* #
from retry import retry
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

# -*-*-*-*-*-*-*-*-*-*-* #
#   Third-Party Modules  #
# -*-*-*-*-*-*-*-*-*-*-* #
import openai
import tiktoken
from openai import OpenAI

class OpenaiAPI:
    def __init__(self, mk1):
        ## System Design
        self.mk1 = mk1

        ## __________ *** Initializing (attributes) *** _______
        self.token_key = str(self.mk1.config.get("api_openai", "token_key"))
        self.model_name = str(self.mk1.config.get("openai", "model_name"))

        ## __________ *** Initializing (client) *** __________
        self.service = self.build_client()

    # Service
    def build_client(self):
        try:
            # Creating the OpenAI API client
            service = OpenAI(api_key = self.token_key)
            self.mk1.logging.logger.info(
                "(OpenaiAPI.build_client) Service build succeeded"
            )
            return service

        except Exception as e:
            self.mk1.logging.logger.error(
                f"(OpenaiAPI.build_client) Service build failed: {e}"
            )
            raise e
            return None


    def generate_summary(
            self, 
            text: str, 
            threshold: int = 7
        ):

        chunks = self._split_into_chunks(
            text = text
        )

        # Processes chunks in parallel
        with ThreadPoolExecutor() as executor:
            responses = list(executor.map(self._generate_summary_chunk, chunks))

        # responses = ".".join(responses)
        # responses = responses.split(".")
        responses = [s for s in responses if len(s.split()) >= threshold]

        return responses

    def _split_into_chunks(self, text: str, tokens: int = 3000):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        words = encoding.encode(text)
        chunks = []
        for i in range(0, len(words), tokens):
            chunks.append(" ".join(encoding.decode(words[i : i + tokens])))
        return chunks


    def _get_prompt(
        self,
        text: str,
        lower_limit_word_count: int = 20,
        upper_limit_word_count: int = 30,
        pct_of_text_to_keep: int = 5,
    ) -> Tuple[str, str]:

        """Determines the appropriate prompt and system content based on the language of the input text.
        Instructions that can be added
        __
        * Strictly keep lanaguge of summary same as language of the provided text.
        * Keep the total number of output words to {pct_of_text_to_keep} % percent of the total number of input text.
        * Keep the total number of output words betweem ({lower_limit_word_count}, {upper_limit_word_count})
        """

        NAME_PREFIX = "Your name is AI player and you are acting"
        PROMPT = (
            NAME_PREFIX
            + f"""As a professional summarizer,
            you have the ability to create a concise and comprehensive summary of the provided text.
            You could detect the language of input text and output the summary in the same language.
            Now you need to follow these guidelines carefully and provide the summary for input text.
            * Language of the summary should always be English, but always show names and namespaces in the original language and English.
            * Craft a concise summary focusing on the most essential information, avoiding unnecessary detail or repetition.
            * Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
            * Rely strictly on the provided text, without including external information.
            * Do not eliminate numbers that are crucial as evidence data.
            * Ensure that the summary is professional and friendly for human reading.
            * Concat similar ideas into a cohesive narrative.
            * Do not use italic, bold, or underline anywhere.
            * Summarize in 1  paragraph and 3 sentence. Maximum of {upper_limit_word_count} words
            * Add 4 topics as hashtags in the end after the summary
            * Add emojis!

            ## Input text: {text}
        """
        )
        return PROMPT


    def _generate_summary_chunk(
            self, 
            text: str,
            temperature : float,
        ):
        """Generates a summary of the input text using the GPT model.

        Args
        ____
            :param: text ('str') - The text to be summarized.
            :param: temperate ('float') - The temprature to be analyzed
            :returns: The generated summary of the input text. If no summary can be generated, returns 'EMPTY'.

        Notes
        _____
            * Summarizes news article titles or similar content in 40 to 50 words.
            * Summaries are returned in paragraph format, not in list format.
            * If a summary cannot be generated, the word 'EMPTY' is returned.
            * https://community.openai.com/t/asking-for-a-summary-of-news-article-titles-and-chat-completion-is-not-able-to-summarise/288015/9

        Raises
        ______
            openai.OpenAIError
                If an error occurs while initializing the OpenAI API or generating a response.
        """

        try:

            prompt = self._get_prompt(
                text=text
            )

            messages = [
                # {'role': 'system', 'content': system_content},
                {"role": "user", "content": prompt}
            ]
            response = self.service.chat.completions.create(
                model       = self.model_name,
                messages    = messages,
                temperature = temperature,
                max_tokens  = 500,
                n           = 1,
                stop        = None,
                # max_tokens       = 300,
                # top_p             = 1,
                # frequency_penalty = 0.0,
                # presence_penalty  = 0.0,
                # stop              = ["\n"]
            )

            self.mk1.logging.logger.info(
                f"(OpenaiAPI.generate_summary_chunk) Summary was generated successfully"
            )
            # Extract the generated content from the response
            return response.choices[0].message.content.strip()

        except openai.OpenAIError as e:
            self.mk1.logging.logger.error(
                f"(OpenaiAPI.generate_summary_chunk) Error occurred while initializing OpenAI API: {e}"
            )
            raise e
            return None


    def analyze_statistical_images(
        self,
        image_paths: List[str],
        analysis_type: str = "general",
        questions_mapping: Dict[str, str] = None,
        confidence_threshold: float = 0.7,
        temperature: float = 0.3
    ):
        """Analyzes a list of statistical images using OpenAI Vision to extract scientific outcomes.
        
        Args:
            image_paths (List[str]): List of paths to statistical image files
            question_mapping (Dict[str, str]): Dictionary mapping question IDs (e.g., 'Q1') to their text representations
            analysis_type (str): Type of analysis to perform ('general', 'correlation', 'regression', 'time_series', etc.)
            confidence_threshold (float): Minimum confidence score to accept results
            temperature (float): Controls randomness in the model's responses (lower = more deterministic)
            
        Returns:
            List[Dict]: List of dictionaries containing analysis results for each image
        """
        self.mk1.logging.logger.info(
            f"(OpenaiAPI.analyze_statistical_images) Analyzing {len(image_paths)} statistical images"
        )
        
        results = []
        
        # Process images in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda path: self._analyze_single_statistical_image(
                    image_path        = path, 
                    questions_mapping = questions_mapping,
                    analysis_type     = analysis_type,
                    temperature       = temperature
                ), 
                image_paths
            ))
        
        # Filter results based on confidence threshold
        validated_results = [
            result for result in results 
            if result and result.get('confidence', 0) >= confidence_threshold
        ]
        
        self.mk1.logging.logger.info(
            f"(OpenaiAPI.analyze_statistical_images) Successfully analyzed {len(validated_results)} out of {len(image_paths)} images"
        )
        
        return validated_results

    def _analyze_single_statistical_image(
            self,
            image_path: str,
            questions_mapping: Dict[str, str] = None,
            analysis_type: str = "general",
            temperature: float = 0.3,
            top_n:int = 3
        ) -> Dict[str, Any]:
        """Analyzes a single statistical image and extracts scientific outcomes.
        
        Args:
            image_path (str): Path to the statistical image file
            analysis_type (str): Type of analysis to perform
            temperature (float): Controls randomness in the model's responses
            
        Returns:
            Dict[str, Any]: Dictionary containing analysis results
        """
        try:
            # Read image file as base64
            import base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Generate appropriate system prompt based on analysis type
            system_prompt = self._get_statistical_analysis_prompt(analysis_type)
            
            # Prepare messages for the API request
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": 
                                f"1. Please analyze this statistical visualization and provide a detailed scientific interpretation of the data presented"
                                f"2. Please take into consideration the questions mapping : {questions_mapping} and replace any abbreviation eg. Q1"
                                f"3. Always explain terms that maybe are not very common eg. 'Sedentary lifestyle' means 'Inactive lifestyle with no exercise"
                                f"4. Extract all numerical values, statistical relationships, and significant findings and then keep the top {top_n} highlights that are worth reading"
                                f"4.1 For the significant findings find really significant, not just obvious observation. Example for this point : "
                                f"4.1.1 Example #1 of significant finding : Physical activity levels show a strong positive correlation with overall health scores (r=0.78, p<0.01)"
                                f"4.1.2 Example #2 - BMI shows a negative correlation with daily step count (r=-0.65, p<0.001)"
                                f"5. For these {top_n} highlights, please present them as they are and then explain them in plain text so that even a kid can understand"
                                f"6. Instead of listing these highlights, please integrate them seamlessly into a well-crafted narrative, as if they were to be published. Use connecting words. Example for this point :"
                                f"6.1 Instead of returning this result (not-desired) :'Q3 (How often do you drink sugary beverages?) has the lowest scores, especially in the high activity group'"
                                f"6.2 You could return (desired) : 'Q3 (How often do you drink sugary beverages?) has the lowest scores, especially in the high activity group, which means probably that the more you exercise the more likely you are prefer more health lifestyle and avoid sugar beverages'"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Make API request
            response = self.service.chat.completions.create(
                model       = "gpt-4o",  # Use vision-capable model
                messages    = messages,
                temperature = temperature,
                max_tokens  = 500,
                n=1
            )
            
            # Process and structure the response
            raw_analysis = response.choices[0].message.content.strip()
            structured_result = self._structure_statistical_analysis(
                raw_analysis  = raw_analysis, 
                analysis_type = analysis_type
            )
            
            # Add metadata
            structured_result['image_path'] = image_path
            structured_result['timestamp'] = datetime.now().isoformat()
            structured_result['confidence'] = 0.9  # Placeholder, would ideally come from model
            
            self.mk1.logging.logger.info(
                f"(OpenaiAPI._analyze_single_statistical_image) Successfully analyzed image: {image_path}"
            )
            
            return structured_result
            
        except Exception as e:
            raise e
            self.mk1.logging.logger.error(
                f"(OpenaiAPI._analyze_single_statistical_image) Error analyzing image {image_path}: {e}"
            )
            return {
                'image_path' : image_path,
                'error'      : str(e),
                'confidence' : 0,
                'timestamp'  : datetime.now().isoformat()
            }

    def _get_statistical_analysis_prompt(self, analysis_type: str) -> str:
        """Returns an appropriate system prompt based on the type of statistical analysis.
        
        Args:
            analysis_type (str): Type of analysis to perform
            
        Returns:
            str: System prompt for the OpenAI API
        """
        prompts = {
            "general": """You are a scientific data analyst specializing in extracting accurate statistical information from visualizations. 
            Analyze the provided statistical image and provide:
            1. Type of visualization (scatter plot, bar chart, histogram, etc.)
            2. Variables shown and their units of measurement
            3. Key statistical findings (mean, median, p-values, confidence intervals, etc.)
            4. Main trends, patterns, or correlations
            5. Statistical significance of findings
            6. Any limitations or potential issues with the visualization
            Format your response as structured information that can be easily parsed.""",
            
            "correlation": """You are a correlation analysis expert. For this statistical visualization:
            1. Identify the correlation coefficient (r or r²) and its value
            2. Describe the strength and direction of the relationship
            3. Note any outliers that may influence the correlation
            4. Mention statistical significance (p-value) if shown
            5. Describe any apparent non-linear relationships
            Format your response as structured information that can be easily parsed.""",
            
            "regression": """You are a regression analysis expert. For this statistical visualization:
            1. Identify the regression equation and coefficients
            2. Note the goodness of fit (R², adjusted R², etc.)
            3. Extract p-values for each coefficient
            4. Describe the significance of each predictor variable
            5. Identify any influential points or outliers
            6. Note any violation of regression assumptions (if visible)
            Format your response as structured information that can be easily parsed.""",
            
            "time_series": """You are a time series analysis expert. For this statistical visualization:
            1. Identify the time period and frequency of measurements
            2. Describe the overall trend (linear, exponential, cyclical, etc.)
            3. Note any seasonality patterns and their period
            4. Identify significant change points or anomalies
            5. Extract growth rates or rates of change
            6. Note any forecasted values and their confidence intervals
            Format your response as structured information that can be easily parsed."""
        }
        
        return prompts.get(analysis_type.lower(), prompts["general"])

    def _structure_statistical_analysis(self, raw_analysis: str, analysis_type: str) -> Dict[str, Any]:
        """Structures the raw analysis text into a standardized dictionary format.
        
        Args:
            raw_analysis (str): Raw text analysis from the OpenAI API
            analysis_type (str): Type of analysis performed
            
        Returns:
            Dict[str, Any]: Structured analysis results
        """
        # Create a basic structure for the results
        result = {
            'analysis_type': analysis_type,
            'raw_analysis': raw_analysis,
            'structured_data': {}
        }
        
        # Use GPT to structure the data by sending another request
        try:
            prompt = f"""
            Parse the following statistical analysis into a structured JSON format:
            
            {raw_analysis}
            
            Extract the following:
            1. visualization_type (string)
            2. variables (list of strings)
            3. statistical_values (dictionary of numerical values)
            4. key_findings (list of strings) 
            5. limitations (list of strings)
            
            Return only the valid JSON with no explanation or additional text.
            """
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self.service.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
                n=1
            )
            
            # Try to parse the response as JSON
            import json
            structured_text = response.choices[0].message.content.strip()
            
            # Clean up the text to ensure it's valid JSON
            # Remove markdown code blocks if present
            if structured_text.startswith("```json"):
                structured_text = structured_text.replace("```json", "").replace("```", "")
            elif structured_text.startswith("```"):
                structured_text = structured_text.replace("```", "")
            
            structured_data = json.loads(structured_text)
            result['structured_data'] = structured_data
            
        except Exception as e:
            self.mk1.logging.logger.warning(
                f"(OpenaiAPI._structure_statistical_analysis) Failed to structure analysis: {e}"
            )
            # If structuring fails, include basic text analysis
            result['structured_data'] = {
                'visualization_type': 'unknown',
                'key_findings': [raw_analysis],
            }
        
        return result


    def _generate_clean_technical_summary(
            self, technical_summary: str,
        ) -> str : 
        """Structures the raw analysis text into a standardized dictionary format.
        
        Args:
            technical_summary (str): Technical summary from .txt
            
        Returns:
            str: Technical summary from .txt well defined
        """
        # Create a basic structure for the results
        
        # Use GPT to structure the data by sending another request
        try:
            prompt = f"""
                Please take the following technical summary provided as a string, and reformat it into a clean, elegant HTML structure with enhanced styling. Follow these guidelines:
                - Remove all empty or null lines to ensure the output is concise and has no extra spacing between elements.
                - Format all section titles (e.g., "1. Strong Relationships between Categorical Variables:") as proper HTML headings with <h3> tags and consistent styling.
                - For each category of text:
                - Display filters at the top inside a visually distinct light gray box using: <div style="background-color:#f5f5f5; padding:12px; border-radius:5px; margin-bottom:15px;">
                - Format filter lists as proper HTML unordered lists with <ul> and <li> tags
                - Present significant findings clearly below the filters with proper HTML formatting and no extra line breaks
                - Ensure the final HTML has proper semantic structure with consistent heading levels
                - Make headlines visually prominent with proper styling (font weight, size, color)
                - Eliminate any unnecessary spacing between elements while maintaining proper readability
                - Maintain a clean, professional appearance with proper HTML structure throughout
                
                The text:  
                    {technical_summary}
            """

            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self.service.chat.completions.create(
                model       = self.model_name,
                messages    = messages,
                temperature = 0.1,
                max_tokens  = 1000,
                n           = 1
            )
            
            # Try to parse the response as JSON
            technical_summary_cleaned = response.choices[0].message.content.strip()
            return technical_summary_cleaned
            
        except Exception as e:
            self.mk1.logging.logger.warning(
                f"(OpenaiAPI._generate_clean_technical_summary) Failed to generate clean technical summary : {e}"
            )
            raise e

    
    def save_report_to_file(
            self,
            html_content: str, 
            output_path: str = "statistical_report.html"
        ):
        """
        Save the HTML report to a file.
        
        Args:
            html_content (str): HTML content of the report
            output_path (str): Path to save the HTML file
            
        Returns:
            str: Path to the saved file
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            return output_path
        except Exception as e:
            print(f"Error saving report: {e}")
            return None
        

    # Example usage:
    def example_usage():
        # Sample analysis results
        sample_results = [
            {
                'image_path': '/path/to/image1.png',
                'confidence': 0.92,
                'structured_data': {
                    'visualization_type': 'Bar Chart',
                    'key_findings': [
                        'Physical activity levels show a strong positive correlation with overall health scores (r=0.78, p<0.01).',
                        'Participants in the high activity group reported 45% fewer health issues than those in the sedentary group.',
                        'Women showed higher adherence to exercise routines (67%) compared to men (51%).'
                    ]
                },
                'raw_analysis': 'This is a bar chart showing the relationship between physical activity and health outcomes...'
            },
            {
                'image_path': '/path/to/image2.png',
                'confidence': 0.85,
                'structured_data': {
                    'visualization_type': 'Scatter Plot',
                    'key_findings': [
                        'BMI shows a negative correlation with daily step count (r=-0.65, p<0.001).',
                        'For every 1000 additional daily steps, BMI decreased by approximately 0.4 points.',
                        'The relationship is non-linear at extreme values, suggesting diminishing returns.'
                    ]
                },
                'raw_analysis': 'The scatter plot demonstrates the relationship between BMI and daily step count...'
            }
        ]

    def generate_statistical_report(
        self,
        analysis_results: List[Dict[str, Any]],
        report_title: str = "Statistical Analysis Report",
        company_name: str = "Data Analysis Team",
        date_format: str = "%B %d, %Y",
        include_raw_analysis: bool = False,
        max_key_findings: int = 5,
        questions_mapping: Dict[str, str] = None,
        technical_summary: str = None
    ) -> str:
        """
        Generate a fancy HTML report summarizing statistical image analyses,
        organized into tabs based on analysis categories.
        
        Args:
            analysis_results (List[Dict]): List of dictionaries containing analysis results for each image
            report_title (str): Title for the report
            company_name (str): Name of the company/team generating the report
            date_format (str): Format for the date in the report
            include_raw_analysis (bool): Whether to include the raw analysis text
            max_key_findings (int): Maximum number of key findings to display per image
            questions_mapping (Dict[str, str]): Mapping of question IDs to their descriptions
            technical_summary (str): Path to a text file containing technical summary content
            
        Returns:
            str: HTML content of the report with tabs
        """
        from datetime import datetime
        import os
        import re
        
        # Get current date
        current_date = datetime.now().strftime(date_format)
        
        # Count total images analyzed
        total_images = len(analysis_results)
        successful_analyses = sum(1 for result in analysis_results if result.get('confidence', 0) >= 0.7)
        
        # Group analyses by category
        categories = {
            'descriptive': [],
            'categorical-categorical': [],
            'categorical-continuous': [],
            'continuous-continuous': []
        }
        
        # Process each analysis result and categorize by folder path
        for analysis in analysis_results:
            image_path = analysis.get('image_path', '')
            
            # Determine category from image path
            category = 'other'
            if '/descriptive/' in image_path:
                category = 'descriptive'
            elif '/categorical-categorical/' in image_path:
                category = 'categorical-categorical'
            elif '/categorical-continuous/' in image_path:
                category = 'categorical-continuous'
            elif '/continuous-continuous/' in image_path:
                category = 'continuous-continuous'
                
            # Add to appropriate category
            if category in categories:
                categories[category].append(analysis)
            else:
                # Handle any uncategorized analyses
                if 'other' not in categories:
                    categories['other'] = []
                categories['other'].append(analysis)
        
        # Create tab names with counts
        tab_names = {
            'technical-summary': "Technical Summary",
            'descriptive': f"Descriptive ({len(categories['descriptive'])})",
            'categorical-categorical': f"Categorical-Categorical ({len(categories['categorical-categorical'])})",
            'categorical-continuous': f"Categorical-Continuous ({len(categories['categorical-continuous'])})",
            'continuous-continuous': f"Continuous-Continuous ({len(categories['continuous-continuous'])})"
        }
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}
        
        # Keep the technical summary in tab_names regardless of whether other categories exist
        category_tab_names = {k: v for k, v in tab_names.items() if k in categories or k == 'technical-summary'}
        tab_names = category_tab_names
        
        # Load technical summary content if provided
        technical_summary_content = ""
        if technical_summary:
            try:
                with open(technical_summary, 'r', encoding='utf-8') as file:
                    technical_summary_content = file.read()
                # Convert plain text to HTML by replacing newlines with <br> tags
                technical_summary_content = technical_summary_content.replace('\n', '<br>')
            except Exception as e:
                technical_summary_content = f"<p>Error loading technical summary: {str(e)}</p>"
        
        # Clean the technical summary (with OpenAI)
        technical_summary_content = self._generate_clean_technical_summary(
            technical_summary = technical_summary_content
        )
        
        # HTML header and styling
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9f9f9;
                }}
                .header {{
                    background: linear-gradient(135deg, #2c3e50, #3498db);
                    color: white;
                    padding: 20px;
                    border-radius: 10px 10px 0 0;
                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                }}
                .header p {{
                    margin: 5px 0 0;
                    font-size: 14px;
                    opacity: 0.8;
                }}
                .report-meta {{
                    background-color: #f5f5f5;
                    border-left: 5px solid #3498db;
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }}
                
                /* Tab styling */
                .tabs {{
                    margin-top: 25px;
                    overflow: hidden;
                    border-radius: 8px 8px 0 0;
                    background-color: #f1f1f1;
                    display: flex;
                    flex-wrap: wrap;
                }}
                .tab {{
                    background-color: inherit;
                    float: left;
                    border: none;
                    outline: none;
                    cursor: pointer;
                    padding: 14px 16px;
                    transition: 0.3s;
                    font-size: 16px;
                    color: #555;
                    border-radius: 8px 8px 0 0;
                    border-right: 1px solid #ddd;
                    flex-grow: 1;
                    text-align: center;
                }}
                .tab:hover {{
                    background-color: #ddd;
                }}
                .tab.active {{
                    background-color: #3498db;
                    color: white;
                }}
                .tabcontent {{
                
                    display: none;
                    padding: 15px;
                    border: 1px solid #ccc;
                    border-top: none;
                    border-radius: 0 0 8px 8px;
                    animation: fadeIn 0.5s;
                    background-color: white;
                }}
                @keyframes fadeIn {{
                    from {{opacity: 0;}}
                    to {{opacity: 1;}}
                }}
                @keyframes fadeEffect {{
                    from {{opacity: 0;}}
                    to {{opacity: 1;}}
                }}
                
                .image-analysis {{
                    background-color: white;
                    margin: 25px 0;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 3px 15px rgba(0, 0, 0, 0.1);
                    transition: all 0.3s ease;
                }}
                .image-analysis:hover {{
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.15);
                    transform: translateY(-5px);
                }}
                .image-display {{
                    margin: 15px 0;
                    text-align: center;
                    border-radius: 5px;
                    overflow: hidden;
                    background-color: #f8f9fa;
                    padding: 10px;
                    border: 1px solid #e9ecef;
                }}
                .analysis-image {{
                    max-width: 100%;
                    height: auto;
                    max-height: 300px;
                    border-radius: 5px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                    cursor: pointer;
                    transition: max-height 0.3s ease;
                }}
                .analysis-image.expanded {{
                    max-height: 600px;
                }}
                .image-controls {{
                    margin-top: 8px;
                    text-align: center;
                }}
                .image-resize-btn {{
                    background-color: #f0f0f0;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px 10px;
                    font-size: 12px;
                    cursor: pointer;
                    color: #555;
                }}
                .image-resize-btn:hover {{
                    background-color: #e0e0e0;
                }}
                
                .image-title {{
                    background-color: #ecf0f1;
                    padding: 12px 20px;
                    border-bottom: 1px solid #ddd;
                    font-weight: 600;
                    color: #2c3e50;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .confidence {{
                    display: inline-block;
                    padding: 3px 10px;
                    border-radius: 15px;
                    font-size: 12px;
                    font-weight: bold;
                }}
                .high-confidence {{
                    background-color: #d4edda;
                    color: #155724;
                }}
                .medium-confidence {{
                    background-color: #fff3cd;
                    color: #856404;
                }}
                .low-confidence {{
                    background-color: #f8d7da;
                    color: #721c24;
                }}
                .analysis-content {{
                    padding: 20px;
                }}
                .key-findings {{
                    background-color: #eaf6ff;
                    border-radius: 5px;
                    padding: 15px;
                    margin-top: 15px;
                    border-left: 4px solid #3498db;
                }}
                .key-findings h4 {{
                    margin-top: 0;
                    color: #2980b9;
                    font-size: 16px;
                }}
                .key-findings ul {{
                    margin-bottom: 0;
                    padding-left: 20px;
                }}
                .key-findings li {{
                    margin-bottom: 8px;
                }}
                .raw-analysis {{
                    font-size: 14px;
                    color: #666;
                    border-top: 1px solid #eee;
                    margin-top: 20px;
                    padding-top: 15px;
                    white-space: pre-line;
                }}
                .visualization-type {{
                    display: inline-block;
                    background-color: #e8f4f8;
                    color: #246;
                    padding: 3px 10px;
                    border-radius: 15px;
                    font-size: 13px;
                    margin-top: 10px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding: 20px;
                    color: #777;
                    font-size: 13px;
                    border-top: 1px solid #eee;
                }}
                .summary-box {{
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 20px 0;
                    border: 1px solid #e9ecef;
                }}
                .category-summary {{
                    background: linear-gradient(to right, #4e54c8, #8f94fb);
                    color: white;
                    border-radius: 12px;
                    padding: 20px 25px;
                    margin: 15px 0 25px 0;
                    box-shadow: 0 4px 12px rgba(78, 84, 200, 0.2);
                    position: relative;
                    overflow: hidden;
                }}

                .category-summary::before {{
                    content: "";
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: url('/api/placeholder/200/200') right center no-repeat;
                    background-size: auto 100%;
                    opacity: 0.15;
                }}

                .category-summary p {{
                    margin: 0;
                    font-size: 16px;
                    font-weight: 500;
                    position: relative;
                    z-index: 2;
                }}

                .category-summary strong {{
                    font-weight: 700;
                    font-size: 20px;
                    display: block;
                    margin-bottom: 6px;
                }}
                
                /* Technical Summary styling */
                #technical-summary {{
                    font-family: 'Inter', sans-serif;
                    max-width: 900px;
                    margin: 0 auto;
                }}
                .technical-summary-content {{
                    background-color: white;
                    padding: 25px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                    font-size: 15px;
                    line-height: 1.5;
                    margin: 20px 0;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }}

                
                .technical-summary-content::before {{
                    content: "📊";
                    position: absolute;
                    top: -12px;
                    left: 20px;
                    background: white;
                    padding: 0 12px;
                    font-size: 22px;
                }}

                .technical-summary-content h3 {{
                    color: #2c3e50;
                    font-size: 18px;
                    font-weight: 600;
                    margin-top: 20px;
                    margin-bottom: 12px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 8px;
                }}

                .technical-summary-content p {{
                    margin-bottom: 15px;
                    color: #333;
                }}

                .technical-summary-content ul {{
                    margin-top: 8px;
                    margin-bottom: 15px;
                    padding-left: 20px;
                }}

                .technical-summary-content li {{
                    margin-bottom: 6px;
                }}

                .technical-summary-content div[style*="background-color:#f5f5f5"] {{
                    margin-bottom: 15px;
                }}

                .technical-summary-content code {{
                    background: #f8f9fa;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-family: 'Consolas', monospace;
                    font-size: 14px;
                    color: #e83e8c;
                }}
                
                /* Questions mapping styling */
                .questions-container {{
                    background: linear-gradient(to right, #f9f9f9, #f1f1f1);
                    border-radius: 10px;
                    padding: 20px;
                    margin: 25px 0;
                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
                }}
                .questions-title {{
                    font-size: 20px;
                    color: #2c3e50;
                    margin-bottom: 15px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 8px;
                }}
                .questions-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 15px;
                }}
                .question-item {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 12px 15px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                    border-left: 3px solid #3498db;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }}
                .question-item:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }}
                .question-id {{
                    font-weight: bold;
                    color: #3498db;
                    margin-right: 8px;
                    font-size: 15px;
                }}
                .question-text {{
                    color: #444;
                }}
                
                @media print {{
                    body {{
                        background-color: white;
                    }}
                    .image-analysis {{
                        break-inside: avoid;
                        box-shadow: none;
                        margin: 15px 0;
                        border: 1px solid #eee;
                    }}
                    .image-analysis:hover {{
                        transform: none;
                        box-shadow: none;
                    }}
                    .header {{
                        background: white;
                        color: black;
                        box-shadow: none;
                        border-bottom: 2px solid #eee;
                    }}
                    .tab {{
                        color: black;
                        background: white !important;
                    }}
                    .tab.active {{
                        border-bottom: 2px solid #3498db;
                    }}
                    .tabcontent {{
                        display: block !important;
                        border: none;
                        page-break-before: always;
                    }}
                    .questions-container {{
                        background: white;
                        box-shadow: none;
                        border: 1px solid #eee;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_title}</h1>
                <p>Generated by {company_name} on {current_date}</p>
            </div>
            
            <div class="report-meta">
                <p><strong>Analysis Summary:</strong> Processed {total_images} statistical images with {successful_analyses} successful analyses ({(successful_analyses/total_images*100) if total_images > 0 else 0:.1f}% success rate)</p>
                <p><strong>Analysis Categories:</strong> {', '.join([v for k, v in tab_names.items() if k != 'technical-summary'])}</p>
            </div>
        """
        
        # Add questions mapping section if provided
        if questions_mapping:
            html_content += """
            <div class="questions-container">
                <div class="questions-title">Survey Questions Reference</div>
                <div class="questions-grid">
            """
            
            # Process each question and replace markdown bold (**) with HTML bold
            for q_id, q_text in questions_mapping.items():
                # Replace markdown bold with HTML bold
                q_text_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', q_text)
                
                html_content += f"""
                <div class="question-item">
                    <span class="question-id">{q_id}:</span>
                    <span class="question-text">{q_text_html}</span>
                </div>
                """
            
            html_content += """
                </div>
            </div>
            """
        
        # Tab navigation
        html_content += """
            <!-- Tab navigation -->
            <div class="tabs">
        """
        
        # Create tab buttons - Technical Summary always first
        tab_order = ['technical-summary'] + [cat for cat in tab_names if cat != 'technical-summary']
        first_tab = True
        
        for category_id in tab_order:
            if category_id in tab_names:
                active_class = " active" if first_tab else ""
                html_content += f'<button class="tab{active_class}" onclick="openTab(event, \'{category_id}\')">{tab_names[category_id]}</button>\n'
                first_tab = False
        
        html_content += """
            </div>
        """

       

        
        
        # Create Technical Summary tab content
        html_content += f"""
            <div id="technical-summary" class="tabcontent" style="display: block;">
                <div class="category-summary">
                    <p><strong>Technical Summary</strong>Detailed technical information about the analysis</p>
                </div>
                <div class="technical-summary-content">
                    {technical_summary_content if technical_summary_content else "<p>No technical summary provided.</p>"}
                </div>
            </div>
        """
        
        # Create tab content for each category
        first_content_tab = False  # Technical Summary is already shown by default
        for category_id, analyses in categories.items():
            if category_id in tab_names:  # Only process categories that have tab names
                display_style = "none"  # All other tabs start hidden
                tab_name = tab_names[category_id]
                
                html_content += f"""
                <div id="{category_id}" class="tabcontent" style="display: {display_style};">
                    <div class="category-summary">
                        <p><strong>{tab_name}</strong>{len(analyses)} analyses ✅</p>
                    </div>
                """
                
                # Add each image analysis within this category
                for index, analysis in enumerate(analyses, 1):
                    # Get image path and format it nicely
                    image_path = analysis.get('image_path', 'Unknown')
                    filename = os.path.basename(image_path) if image_path != 'Unknown' else 'Unknown'
                    
                    # Get analysis data
                    confidence = analysis.get('confidence', 0)
                    confidence_class = 'high-confidence' if confidence >= 0.85 else 'medium-confidence' if confidence >= 0.7 else 'low-confidence'
                    
                    # Get structured data
                    structured_data = analysis.get('structured_data', {})
                    visualization_type = structured_data.get('visualization_type', 'Unknown')
                    
                    # Get key findings
                    key_findings = structured_data.get('key_findings', [])
                    if isinstance(key_findings, str):
                        key_findings = [key_findings]
                    
                    # Limit key findings to max_key_findings
                    key_findings = key_findings[:max_key_findings]
                    
                    # Get raw analysis
                    raw_analysis = analysis.get('raw_analysis', '')
                    
                    # Create the analysis section
                    html_content += f"""
                    <div class="image-analysis">
                        <div class="image-title">
                            Image {index}: {filename}
                            <span class="confidence {confidence_class}">{confidence*100:.1f}% Confidence</span>
                        </div>
                        <div class="analysis-content">
                            <div class="image-display">
                                <img src="{image_path}" alt="{filename}" class="analysis-image" onclick="toggleImageSize(this)">
                                <div class="image-controls">
                                    <button onclick="toggleImageSize(this.parentElement.previousElementSibling)" class="image-resize-btn">Resize</button>
                                </div>
                            </div>
                            <div class="visualization-type">{visualization_type} Visualization</div>
                            
                            <div class="key-findings">
                                <h4>Key Findings:</h4>
                                <ul>
                    """
                    
                    # Add key findings
                    for finding in key_findings:
                        html_content += f"<li>{finding}</li>\n"
                    
                    # If no key findings, add a message
                    if not key_findings:
                        html_content += "<li>No significant findings detected.</li>\n"
                    
                    html_content += """
                                </ul>
                            </div>
                    """
                    
                    # Add raw analysis if requested
                    if include_raw_analysis and raw_analysis:
                        # Limit to 500 characters with ellipsis for brevity
                        display_raw = raw_analysis[:500] + ("..." if len(raw_analysis) > 500 else "")
                        html_content += f"""
                            <div class="raw-analysis">
                                <details>
                                    <summary>Raw Analysis</summary>
                                    {display_raw}
                                </details>
                            </div>
                        """
                    
                    html_content += """
                        </div>
                    </div>
                    """
                
                html_content += """
                </div>
                """
                
                first_content_tab = False
        
        # Add footer and JavaScript
        html_content += f"""
            <div class="footer">
                <p>This report was automatically generated on {current_date}. The analysis was performed using advanced computer vision and natural language processing techniques.</p>
                <p>&copy; {datetime.now().year} {company_name}. All rights reserved.</p>
            </div>
            <script>
                function openTab(evt, categoryId) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].style.display = "none";
                    }}
                    tablinks = document.getElementsByClassName("tab");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    document.getElementById(categoryId).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
                
                function toggleImageSize(img) {{
                    img.classList.toggle('expanded');
                }}
                
                // Ensure the first tab is active by default
                document.addEventListener('DOMContentLoaded', function() {{
                    // Get the first tab and click it
                    var firstTab = document.querySelector('.tab');
                    if (firstTab) {{
                        firstTab.click();
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        return html_content
        