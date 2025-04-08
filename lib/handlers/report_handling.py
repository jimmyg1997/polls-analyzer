
import os
import argparse
import tempfile
import shutil
import re
import asyncio
import pandas as pd
from datetime import datetime
from pyppeteer import launch
from typing import Dict, Any, List


class ReportHandler:
    def __init__(self, mk1, google_email_api, openai_api):
        self.mk1 = mk1
        self.google_email_api = google_email_api
        self.openai_api = openai_api

    async def convert_html_to_pdf(self, html_path, pdf_path=None, wait_time=2000, include_styles=True):
        """
        Convert HTML file with dynamic tabs to PDF, ensuring all tabbed content is visible.
        
        Parameters:
        - html_path: Path to the HTML file
        - pdf_path: Output PDF path (if None, will use the same name as HTML but with .pdf extension)
        - wait_time: Time to wait in ms for JavaScript to execute
        - include_styles: Whether to include extra print styles to improve PDF output
        
        Returns:
        - Path to the generated PDF file
        """
        # Set default PDF path if not provided
        if not pdf_path:
            pdf_path = os.path.splitext(html_path)[0] + '.pdf'
        
        # Create a temporary file with modified HTML if needed
        temp_dir = None
        temp_html_path = html_path
        
        try:
            # Always use temp dir to handle image paths
            temp_dir = tempfile.mkdtemp()
            temp_html_path = os.path.join(temp_dir, os.path.basename(html_path))
            
            # Get the directory of the original HTML file for image resolution
            html_dir = os.path.dirname(os.path.abspath(html_path))
            
            # Read the HTML file
            with open(html_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            # Fix image paths to absolute paths
            def fix_image_path(match):
                img_path = match.group(1)
                
                # Skip URLs and data URIs
                if img_path.startswith(('http://', 'https://', 'data:')):
                    return f'src="{img_path}"'
                
                # Handle relative paths
                if img_path.startswith('./'):
                    img_path = img_path[2:]  # Remove leading ./
                
                # Create absolute path
                abs_img_path = os.path.normpath(os.path.join(html_dir, img_path))
                
                # Verify the image exists
                if os.path.exists(abs_img_path):
                    # Use the file:// protocol for absolute paths
                    return f'src="file://{abs_img_path}"'
                else:
                    print(f"Warning: Image not found at {abs_img_path}")
                    return f'src="{img_path}"'  # Keep original if not found
            
            # Replace image paths in HTML
            html_content = re.sub(r'src="([^"]+)"', fix_image_path, html_content)
            
            # Add print-specific CSS if requested
            if include_styles:
                print_css = """
                    @media print {
                        /* Force display all tabs in print mode */
                        .tabcontent {
                            display: block !important;
                            page-break-before: always;
                            margin-top: 20px;
                            padding-top: 20px;
                            border-top: 2px solid #000;
                        }
                        
                        /* Make sure tab headers appear properly */
                        .tab {
                            color: #000 !important;
                            background: #f0f0f0 !important;
                            font-weight: bold;
                            border: 1px solid #ccc !important;
                        }
                        
                        /* Add tab titles before content for clarity */
                        .tabcontent::before {
                            content: attr(id);
                            display: block;
                            font-size: 24px;
                            font-weight: bold;
                            margin-bottom: 15px;
                            text-transform: capitalize;
                        }
                        
                        /* Optimize images */
                        img {
                            max-width: 90% !important;
                            height: auto !important;
                            margin: 0 auto !important;
                            display: block !important;
                        }
                        
                        /* Headers should be clearly visible */
                        .header {
                            background: #f0f0f0 !important; 
                            color: #000 !important;
                            padding: 15px !important;
                            margin-bottom: 20px !important;
                        }
                    }
                """
                
                # Add print styles to the HTML head
                if '</head>' in html_content:
                    html_content = html_content.replace('</head>', f'<style>{print_css}</style></head>')
                else:
                    html_content = f'<style>{print_css}</style>{html_content}'
            
            # Write the modified HTML to the temporary file
            with open(temp_html_path, 'w', encoding='utf-8') as file:
                file.write(html_content)
            
            # Now handle any static assets needed
            # Create a directory structure for static files in the temp dir if needed
            static_dir = os.path.join(html_dir, 'static')
            if os.path.exists(static_dir):
                temp_static_dir = os.path.join(temp_dir, 'static')
                os.makedirs(temp_static_dir, exist_ok=True)
                
                # Create subdirectories as needed for FFQ folder and its subfolders
                ffq_dir = os.path.join(static_dir, 'FFQ')
                if os.path.exists(ffq_dir):
                    temp_ffq_dir = os.path.join(temp_static_dir, 'FFQ')
                    os.makedirs(temp_ffq_dir, exist_ok=True)
                    
                    # Copy all the subdirectories of FFQ
                    for subdir in ['descriptive', 'categorical-categorical', 'categorical-continuous', 'continuous-continuous']:
                        src_subdir = os.path.join(ffq_dir, subdir)
                        if os.path.exists(src_subdir):
                            dst_subdir = os.path.join(temp_ffq_dir, subdir)
                            os.makedirs(dst_subdir, exist_ok=True)
                            
                            # Copy all images in this subdirectory
                            for file_name in os.listdir(src_subdir):
                                if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                                    src_file = os.path.join(src_subdir, file_name)
                                    dst_file = os.path.join(dst_subdir, file_name)
                                    shutil.copy2(src_file, dst_file)
            
            # Get absolute file path for the HTML
            absolute_path = 'file://' + os.path.abspath(temp_html_path)
            
            # Launch browser
            browser = await launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
            page = await browser.newPage()
            
            # Set viewport for better rendering
            await page.setViewport({'width': 1200, 'height': 1600})
            
            # Navigate to the HTML file
            await page.goto(absolute_path, {'waitUntil': 'networkidle0'})
            
            # Wait for JavaScript to execute
            await page.waitFor(wait_time)
            
            # Execute custom JavaScript to prepare tabs for PDF rendering
            await page.evaluate('''() => {
                // Get all tabcontent divs
                const tabContents = document.querySelectorAll('.tabcontent');
                
                // First, ensure each tab has an accessible title by using its ID
                tabContents.forEach(content => {
                    // Convert ID like 'technical-summary' to 'Technical Summary'
                    const title = content.id
                        .split('-')
                        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                        .join(' ');
                    
                    // Add title as data attribute for CSS ::before content
                    content.setAttribute('data-title', title);
                });
                
                // Make sure all tabs are forced visible
                tabContents.forEach(content => {
                    content.style.display = 'block';
                    content.style.opacity = '1';
                    content.style.visibility = 'visible';
                    content.style.pageBreakBefore = 'always';
                });
                
                // Make sure tab buttons visuals are preserved
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => {
                    // Keep the active tab styling on its button
                    if (tab.classList.contains('active')) {
                        tab.style.backgroundColor = '#3498db';
                        tab.style.color = 'white';
                    } else {
                        tab.style.backgroundColor = '#f1f1f1';
                        tab.style.color = '#555';
                    }
                });

                // Ensure images are fully loaded and sized appropriately
                const images = document.querySelectorAll('img');
                images.forEach(img => {
                    if (img.classList.contains('analysis-image')) {
                        // Ensure image is visible and properly sized
                        img.style.maxHeight = 'none';
                        img.style.height = 'auto';
                        img.style.display = 'block';
                        img.style.margin = '0 auto';
                        img.style.maxWidth = '90%';
                    }
                    
                    // Check if image loaded correctly
                    if (img.complete && img.naturalHeight === 0) {
                        console.log('Image failed to load:', img.src);
                        // Optionally add a placeholder or error message
                        img.style.border = '2px dashed red';
                        img.style.padding = '20px';
                        img.style.width = '300px';
                        img.style.height = '200px';
                        img.style.background = '#f8f8f8';
                        img.style.display = 'flex';
                        img.style.alignItems = 'center';
                        img.style.justifyContent = 'center';
                        
                        // Create text node for error message
                        const errorText = document.createElement('div');
                        errorText.textContent = 'Image not found: ' + img.alt;
                        errorText.style.textAlign = 'center';
                        errorText.style.color = 'red';
                        errorText.style.fontWeight = 'bold';
                        
                        // Insert error message after image
                        img.parentNode.insertBefore(errorText, img.nextSibling);
                    }
                });
            }''')
            
            # Wait again to make sure changes apply
            await page.waitFor(1000)
            
            # Generate PDF
            await page.pdf({
                'path': pdf_path,
                'format': 'A4',
                'printBackground': True,
                'preferCSSPageSize': True,
                'margin': {'top': '20mm', 'right': '20mm', 'bottom': '20mm', 'left': '20mm'}
            })
            
            # Close browser
            await browser.close()
            
            return pdf_path
            
        finally:
            # Clean up temporary directory if created
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def html_to_pdf(self, html_path, pdf_path=None, wait_time=2000, include_styles=True) -> str:
        """Synchronous wrapper for the async convert function"""
        return asyncio.get_event_loop().run_until_complete(
            self.convert_html_to_pdf(html_path, pdf_path, wait_time, include_styles)
        )

    def process_dir(self, directory, output_dir=None, wait_time=2000, include_styles=True):
        """Process all HTML files in a directory"""
        if not output_dir:
            output_dir = directory
        
        os.makedirs(output_dir, exist_ok=True)
        
        html_files = [f for f in os.listdir(directory) if f.endswith('.html')]
        results = []
        
        for html_file in html_files:
            html_path = os.path.join(directory, html_file)
            pdf_path = os.path.join(output_dir, os.path.splitext(html_file)[0] + '.pdf')
            
            try:
                result = self.html_to_pdf(html_path, pdf_path, wait_time, include_styles)
                results.append((html_file, "Success", result))
                print(f"Converted {html_file} to {pdf_path}")
            except Exception as e:
                results.append((html_file, "Failed", str(e)))
                print(f"Failed to convert {html_file}: {str(e)}")
        
        return results

    
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
        

    def generate_statistical_report(
        self,
        analysis_results: List[Dict[str, Any]],
        report_title: str = "Statistical Analysis Report",
        questionnaire_name: str = "FFQ",
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
            'technical-summary'       : "Technical Summary",
            'descriptive'             : f"Descriptive ({len(categories['descriptive'])})",
            'categorical-categorical' : f"Categorical-Categorical ({len(categories['categorical-categorical'])})",
            'categorical-continuous'  : f"Categorical-Continuous ({len(categories['categorical-continuous'])})",
            'continuous-continuous'   : f"Continuous-Continuous ({len(categories['continuous-continuous'])})"
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
        technical_summary_content = self.openai_api._generate_clean_technical_summary(
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
                    # TODO : find it more dynamically!
                    image_path = analysis.get('image_path', 'Unknown')
                    image_path = image_path.replace(f"./static/{questionnaire_name}", ".")
                    
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

    def send_report_by_email(
            self, 
            report_path              : str, 
            df                       : pd.DataFrame, 
            cc                       : str  = None,
            subject                  : str  = None, 
            fn_non_technical_summary : str  = None,
            message                  : str  = None, 
            pdf_conversion           : bool = False
        ):
        """
        Send HTML or PDF reports via email to recipients.
        
        Parameters:
            - report_path: Path to the HTML or PDF report
            - df: DataFrame that contains an 'Email' column with recipient addresses
            - cc: Email addresses to carbon copy (optional)
            - subject: Email subject (if None, a default will be used)
            - fn_non_technical_summary: Path to a text file containing a non-technical summary
            - message: Email body text (if None, a default will be used)
            - pdf_conversion: If True and report is HTML, convert to PDF before sending
            
        Returns:
            - Dictionary with results for each recipient: {email: status}
        """
        # Input validation
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Report file not found: {report_path}")
        
        if 'Email' not in df.columns:
            raise ValueError("DataFrame must contain an 'Email' column")
        
        # Get unique email addresses
        unique_emails = [
            email.lower() for email in list(df['Email'].unique()) if email and "@" in email
        ]

        if len(unique_emails) == 0:
            return {"error": "No email addresses found in DataFrame"}
        
        # Set default subject and message if not provided
        if not subject:
            report_name = os.path.basename(report_path)
            subject = f"Your Report: {report_name}"
        
        if not message:
            message = """ The results are in! 📊✨
            \nWe've officially completed the full analysis of your survey responses, and the detailed findings are now available — check out the attached PDF! 📎📄
            \nThis report is the result of comprehensive statistical analysis 🧠📈 across all participants — both anonymous and identified. All graphs 📊 were auto-generated, and the statistical tests 🧪 were performed automatically for accuracy and transparency.
            \nTo take things a step further, we even asked OpenAI 🤖 to interpret our statistical visuals and highlight the most interesting insights 🔍💡😉
            \nThis project was led by me, Dimitrios Georgiou, a 𝐃𝐚𝐭𝐚 𝐒𝐜𝐢𝐞𝐧𝐭𝐢𝐬𝐭 🧑‍💻📚, aiming to explore behavioral patterns and lifestyle trends based on your input. The data was treated with care, kept secure 🔐, and used solely for research purposes.
            \nThank you all for participating 🙌 Your contributions help us understand the world just a bit better 🌍❤️
            \n👉 𝘗𝘭𝘦𝘢𝘴𝘦 𝘭𝘦𝘢𝘷𝘦 𝘺𝘰𝘶𝘳 𝘧𝘦𝘦𝘥𝘣𝘢𝘤𝘬 𝘣𝘺 𝘴𝘪𝘮𝘱𝘭𝘺 𝘳𝘦𝘱𝘭𝘺𝘪𝘯𝘨 𝘵𝘰 𝘵𝘩𝘪𝘴 𝘦𝘮𝘢𝘪𝘭! 💬📩 𝘦𝘨. 𝘛𝘩𝘦 𝘢𝘴𝘴𝘰𝘤𝘪𝘢𝘵𝘪𝘰𝘯𝘴 𝘢𝘳𝘦 𝘯𝘰𝘵 𝘸𝘦𝘭𝘭 𝘦𝘹𝘱𝘭𝘢𝘪𝘯𝘦𝘥, 𝘰𝘳 𝘵𝘩𝘦𝘳𝘦 𝘢𝘳𝘦 𝘰𝘷𝘦𝘳𝘭𝘢𝘱𝘴 𝘦𝘵𝘤
        """
        
        # Read and incorporate non-technical summary if provided
        if fn_non_technical_summary and os.path.exists(fn_non_technical_summary):
            try:
                with open(fn_non_technical_summary, 'r', encoding='utf-8') as f:
                    summary_content = f.read()
                    
                # Format the summary in a more visually appealing way
                formatted_summary = self._format_summary_for_email(summary_content)
                
                # Append the formatted summary to the message
                message += "\n\n" + "=" * 50 + "\n\n"
                message += "📋 𝗦𝗨𝗠𝗠𝗔𝗥𝗬 𝗢𝗙 𝗞𝗘𝗬 𝗙𝗜𝗡𝗗𝗜𝗡𝗚𝗦 📋\n\n"
                message += formatted_summary
                message += "\n\n" + "=" * 50 + "\n\n"
                message += "For complete details, please refer to the attached report. 🔍"
                
            except Exception as e:
                print(f"Warning: Could not read summary file: {e}")
        
        # Check if we need to convert HTML to PDF
        temp_pdf_path = None
        attachment_path = report_path
        
        if pdf_conversion and report_path.lower().endswith('.html'):
            try:
                # Use the existing conversion method
                temp_pdf_path = os.path.splitext(report_path)[0] + '.pdf'
                attachment_path = self.html_to_pdf(report_path, temp_pdf_path)
            except Exception as e:
                return {"error": f"Failed to convert HTML to PDF: {str(e)}"}
        
        # Determine attachment type
        is_pdf = attachment_path.lower().endswith('.pdf')
        content_type = "application/pdf" if is_pdf else "text/html"
        
        # Results tracking
        results = {}
        
        try:
            # For each recipient
            for email in unique_emails:
                if not email or not isinstance(email, str) or '@' not in email:
                    results[str(email)] = "Invalid email address"
                    continue
                    
                try:
                    # Use the google_email_api to send the email with attachment
                    email_status = self.google_email_api.send_email(
                        recipient       = email,
                        cc              = cc,
                        subject         = subject,
                        body_text       = message,
                        attachment_path = attachment_path,
                        attachment_type = content_type
                    )
                    
                    # Track results
                    results[email] = "Success" if email_status else "Failed"
                    
                except Exception as e:
                    results[email] = f"Error: {str(e)}"
            
            return results
            
        finally:
            # Clean up temporary PDF if created
            if temp_pdf_path and os.path.exists(temp_pdf_path) and temp_pdf_path != report_path:
                try:
                    os.remove(temp_pdf_path)
                except:
                    pass

    def _format_summary_for_email(self, summary_text):
        """
        Format the summary text to make it more visually appealing in an email.
        
        Args:
            summary_text (str): Raw summary text from file
            
        Returns:
            str: Formatted summary text with enhanced styling
        """
        # Split the summary into lines
        lines = summary_text.strip().split('\n')
        
        # Format the title if it exists
        formatted_text = ""
        
        # Create a dictionary of emoji mappings for different categories
        emoji_map = {
            "total responses": "👥",
            "gender": "⚧️",
            "city": "🏙️",
            "completion time": "⏱️",
            "highest scoring": "📈",
            "lowest scoring": "📉",
            "age": "🗓️",
            "education": "🎓",
            "employment": "💼",
            "physical activity": "🏃‍♀️"
        }
        
        # Process each bullet point with appropriate emoji
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('•'):
                line = line[1:].strip()
                
            # Find appropriate emoji
            emoji = "📊"  # Default emoji
            
            # Check for both highest and lowest scoring questions
            if "highest scoring" in line.lower():
                emoji = emoji_map["highest scoring"]
            elif "lowest scoring" in line.lower() or "howest scoring" in line.lower():
                emoji = emoji_map["lowest scoring"]
            else:
                # Check other categories
                for key, custom_emoji in emoji_map.items():
                    if key in line.lower():
                        emoji = custom_emoji
                        break
                    
            # Format with emoji and styling
            formatted_text += f"{emoji} {line}\n"
        
        return formatted_text