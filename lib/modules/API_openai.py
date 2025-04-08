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
import os
import re
import time
import random
import threading
import concurrent.futures

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

class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, tokens_per_second: float):
        """
        Initialize a rate limiter.
        
        Args:
            tokens_per_second (float): Tokens added per second
        """
        self.tokens_per_second = tokens_per_second
        self.token_bucket = 1.0  # Start with one token
        self.max_tokens = 1.0    # Maximum number of tokens
        self.last_update = time.time()
        self.lock = threading.Lock()
        
    def acquire(self):
        """
        Acquires a token from the bucket, blocking if necessary.
        """
        with self.lock:
            current_time = time.time()
            # Add tokens based on elapsed time
            time_elapsed = current_time - self.last_update
            new_tokens = time_elapsed * self.tokens_per_second
            self.token_bucket = min(self.max_tokens, self.token_bucket + new_tokens)
            self.last_update = current_time
            
            # If no tokens available, sleep until one is available
            if self.token_bucket < 1.0:
                sleep_time = (1.0 - self.token_bucket) / self.tokens_per_second
                time.sleep(sleep_time)
                self.token_bucket = 0.0
                self.last_update = time.time()
            else:
                self.token_bucket -= 1.0

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


    # def analyze_statistical_images(
    #     self,
    #     image_paths: List[str],
    #     analysis_type: str = "general",
    #     questions_mapping: Dict[str, str] = None,
    #     confidence_threshold: float = 0.7,
    #     temperature: float = 0.3
    # ):
    #     """Analyzes a list of statistical images using OpenAI Vision to extract scientific outcomes.
        
    #     Args:
    #         image_paths (List[str]): List of paths to statistical image files
    #         question_mapping (Dict[str, str]): Dictionary mapping question IDs (e.g., 'Q1') to their text representations
    #         analysis_type (str): Type of analysis to perform ('general', 'correlation', 'regression', 'time_series', etc.)
    #         confidence_threshold (float): Minimum confidence score to accept results
    #         temperature (float): Controls randomness in the model's responses (lower = more deterministic)
            
    #     Returns:
    #         List[Dict]: List of dictionaries containing analysis results for each image
    #     """
    #     self.mk1.logging.logger.info(
    #         f"(OpenaiAPI.analyze_statistical_images) Analyzing {len(image_paths)} statistical images"
    #     )
        
    #     results = []
        
    #     # Process images in parallel
    #     with ThreadPoolExecutor() as executor:
    #         results = list(executor.map(
    #             lambda path: self._analyze_single_statistical_image(
    #                 image_path        = path, 
    #                 questions_mapping = questions_mapping,
    #                 analysis_type     = analysis_type,
    #                 temperature       = temperature
    #             ), 
    #             image_paths
    #         ))
        
    #     # Filter results based on confidence threshold
    #     validated_results = [
    #         result for result in results 
    #         if result and result.get('confidence', 0) >= confidence_threshold
    #     ]
        
    #     self.mk1.logging.logger.info(
    #         f"(OpenaiAPI.analyze_statistical_images) Successfully analyzed {len(validated_results)} out of {len(image_paths)} images"
    #     )
        
    #     return validated_results

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

        print(image_path)
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


    def analyze_statistical_images(
            self,
            image_paths: List[str],
            analysis_type: str = "general",
            questions_mapping: Dict[str, str] = None,
            confidence_threshold: float = 0.7,
            temperature: float = 0.3,
            max_requests_per_minute: int = 25,  # Lower than the actual limit to be safe
            max_workers: int = 5,  # Limit concurrent threads
            retry_delay: float = 2.0  # Seconds to wait after a rate limit error
        ):
            """Analyzes a list of statistical images using OpenAI Vision to extract scientific outcomes.
            
            Args:
                image_paths (List[str]): List of paths to statistical image files
                questions_mapping (Dict[str, str]): Dictionary mapping question IDs to their text representations
                analysis_type (str): Type of analysis to perform ('general', 'correlation', 'regression', etc.)
                confidence_threshold (float): Minimum confidence score to accept results
                temperature (float): Controls randomness in the model's responses (lower = more deterministic)
                max_requests_per_minute (int): Maximum number of API requests per minute
                max_workers (int): Maximum number of concurrent worker threads
                retry_delay (float): Seconds to wait after encountering a rate limit error
                
            Returns:
                List[Dict]: List of dictionaries containing analysis results for each image
            """
            self.mk1.logging.logger.info(
                f"(OpenaiAPI.analyze_statistical_images) Analyzing {len(image_paths)} statistical images"
            )
            
            # Create a token bucket for rate limiting
            rate_limiter = RateLimiter(max_requests_per_minute / 60.0)  # Tokens per second
            
            # Process images with controlled parallelism
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {
                    executor.submit(
                        self._analyze_single_statistical_image_with_rate_limit,
                        image_path=path,
                        questions_mapping=questions_mapping,
                        analysis_type=analysis_type,
                        temperature=temperature,
                        rate_limiter=rate_limiter,
                        retry_delay=retry_delay
                    ): path for path in image_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.mk1.logging.logger.error(
                            f"(OpenaiAPI.analyze_statistical_images) Error processing image {path}: {e}"
                        )
                        # Add error entry to results
                        results.append({
                            'image_path': path,
                            'error': str(e),
                            'confidence': 0,
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Filter results based on confidence threshold
            validated_results = [
                result for result in results 
                if result and result.get('confidence', 0) >= confidence_threshold
            ]
            
            self.mk1.logging.logger.info(
                f"(OpenaiAPI.analyze_statistical_images) Successfully analyzed {len(validated_results)} out of {len(image_paths)} images"
            )
            
            return validated_results

    def _analyze_single_statistical_image_with_rate_limit(
            self,
            image_path: str,
            questions_mapping: Dict[str, str],
            analysis_type: str,
            temperature: float,
            rate_limiter,
            retry_delay: float = 2.0,
            max_retries: int = 5
        ) -> Dict[str, Any]:
        """Analyzes a single statistical image with rate limiting and retries.
        
        Args:
            image_path (str): Path to the statistical image file
            questions_mapping (Dict[str, str]): Dictionary mapping question IDs to their text
            analysis_type (str): Type of analysis to perform
            temperature (float): Controls randomness in the model's responses
            rate_limiter: Rate limiter instance to control API call frequency
            retry_delay (float): Base delay for retries in seconds
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Dict[str, Any]: Dictionary containing analysis results
        """
        print(image_path)
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Wait for token from rate limiter before making request
                rate_limiter.acquire()
                
                # Perform the analysis
                result = self._analyze_single_statistical_image(
                    image_path        = image_path,
                    questions_mapping = questions_mapping,
                    analysis_type     = analysis_type,
                    temperature       = temperature
                )

                print(f"[Success] For {image_path}")
                return result
                
            except openai.RateLimitError as e:
                print(f"[Failure] For {image_path} : {e}")
                retry_count += 1
                self.mk1.logging.logger.warning(
                    f"(OpenaiAPI._analyze_single_statistical_image_with_rate_limit) "
                    f"Rate limit error on image {image_path}, retry {retry_count}/{max_retries}: {e}"
                )
                
                # Exponential backoff with jitter for retries
                wait_time = retry_delay * (2 ** retry_count) + random.uniform(0, 1)
                time.sleep(wait_time)
                
        # If we've exhausted retries, raise the last exception
        self.mk1.logging.logger.error(
            f"(OpenaiAPI._analyze_single_statistical_image_with_rate_limit) "
            f"Failed to process image {image_path} after {max_retries} retries"
        )
        
        return {
            'image_path': image_path,
            'error': f"Failed after {max_retries} retries",
            'confidence': 0,
            'timestamp': datetime.now().isoformat()
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

    
        