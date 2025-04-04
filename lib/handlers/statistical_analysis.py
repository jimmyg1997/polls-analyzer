# basics
import os
import ast
import itertools
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple
from matplotlib.patches import Patch
from datetime import datetime


# visualize
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
from PIL import Image, ImageDraw, ImageFont
        

# apis
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LogisticRegression
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.proportion import proportion_confint
import pingouin as pg


class StatisticalTests:
    def __init__(self, significance_threshold=0.05):
        self.significance_threshold = significance_threshold


    def get_all_visualization_paths(
            self, 
            dir_static             : str,
            questionnaire_name     : str
        ) : 
        chi2_results = pd.read_csv(
            filepath_or_buffer = f"{dir_static}/{questionnaire_name}/categorical-categorical/chi2_results.csv", 
        )

        anova_results = pd.read_csv(
            filepath_or_buffer = f"{dir_static}/{questionnaire_name}/categorical-continuous/anova_results.csv", 
        )
        nonparam_group_results = pd.read_csv(
            filepath_or_buffer = f"{dir_static}/{questionnaire_name}/categorical-continuous/nonparametric_results.csv", 
        )
        pearson_results = pd.read_csv(
            filepath_or_buffer = f"{dir_static}/{questionnaire_name}/continuous-continuous/pearson_results.csv", 
        )
        spearman_results = pd.read_csv(
            filepath_or_buffer = f"{dir_static}/{questionnaire_name}/continuous-continuous/spearman_results.csv", 
        )

        strong_findings = {
            "categorical_categorical": chi2_results[
                (chi2_results["Significant"] == True) & 
                (chi2_results["Passed_All_Filters"] == True)
                #(chi2_results["Passed_Any_Filter"] == True if "Passed_Any_Filter" in chi2_results.columns else True)
            ],
            "categorical_continuous_parametric": anova_results[
                (anova_results["Significant"] == True) & 
                (anova_results["Passed_All_Filters"] == True)
            ],
            "categorical_continuous_nonparametric": nonparam_group_results[
                (nonparam_group_results["Significant"] == True) & 
                (nonparam_group_results["Passed_All_Filters"] == True)
            ],
            "continuous_continuous_parametric": pearson_results[
                (pearson_results["Significant"] == True) & 
                (pearson_results["Passed_All_Filters"] == True)
            ],
            "continuous_continuous_nonparametric": spearman_results[
                (spearman_results["Significant"] == True) & 
                (spearman_results["Passed_All_Filters"] == True)
            ]
        }

        overall_analysis_path         = f"{dir_static}/{questionnaire_name}/significant_findings_summary.txt"
        descriptive_paths             = os.listdir(f"{dir_static}/{questionnaire_name}/descriptive")
        categorical_categorical_paths = os.listdir(f"{dir_static}/{questionnaire_name}/categorical-categorical")
        categorical_continuous_paths  = os.listdir(f"{dir_static}/{questionnaire_name}/categorical-continuous")
        continuous_continuous_paths   = os.listdir(f"{dir_static}/{questionnaire_name}/continuous-continuous")

        strong_findings_paths = {
            "overall_analysis_path"                : overall_analysis_path,
            "descriptive"                          : [f"{dir_static}/{questionnaire_name}/descriptive/{fn}" for fn in descriptive_paths],
            "categorical_categorical"              : [f"{dir_static}/{questionnaire_name}/categorical-categorical/{fn}" for fn in categorical_categorical_paths],
            "categorical_continuous_parametric"    : [f"{dir_static}/{questionnaire_name}/categorical-continuous/{fn}" for fn in categorical_continuous_paths if 'nonparametric' not in fn],
            "categorical_continuous_nonparametric" : [f"{dir_static}/{questionnaire_name}/categorical-continuous/{fn}" for fn in categorical_continuous_paths if 'nonparametric' in fn],
            "continuous_continuous_nonparametric"  : [f"{dir_static}/{questionnaire_name}/continuous-continuous/{fn}" for fn in continuous_continuous_paths],
        }

        ## Utility Function
        def filter_list(file_list, df, ft1, ft2):
            filtered_list = []
            for _, row in df.iterrows():
                fn_prefix = f"{row[f'{ft1}']}_{row[f'{ft2}']}"
                filtered_sublist = [fn_path for fn_path in file_list if fn_prefix in fn_path]
                filtered_list += filtered_sublist
            return filtered_list
    

        def txt_to_png(input_txt, font_size=12):
            with open(input_txt, 'r') as f:
                text = f.read()
            
            # Use monospaced font
            try:
                font = ImageFont.truetype("cour.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate dimensions
            lines = text.split('\n')
            dummy_img = Image.new('RGB', (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_img)
            
            # Get accurate line height with some extra spacing
            bbox = dummy_draw.textbbox((0, 0), 'Ay', font=font)  # Using 'Ay' to get full height including descenders
            line_height = int((bbox[3] - bbox[1]) * 1.2)  # 20% extra spacing
            max_width = max(font.getlength(line) for line in lines) if lines else 0
            
            # Increase padding
            padding = 20
            img_width = int(max_width + padding * 2)
            img_height = int(line_height * len(lines) + padding * 2)
            
            # Create image
            img = Image.new('RGB', (img_width, img_height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw each line individually to ensure proper spacing
            y_pos = padding
            for line in lines:
                draw.text((padding, y_pos), line, font=font, fill='black')
                y_pos += line_height
            
            # Save and return
            output_path = os.path.join(os.path.dirname(input_txt), "significant_findings_summary.png")
            img.save(output_path)
            return output_path


        ## Overall Analysis (.txt -> .png)
        strong_findings_paths["overall_analysis_path"] = txt_to_png(
            input_txt = strong_findings_paths["overall_analysis_path"]
        )

        ## 1. Strong Relationships between Categorical Variables
        strong_findings_paths["categorical_categorical"] = filter_list(
            file_list = strong_findings_paths["categorical_categorical"],
            df        = strong_findings["categorical_categorical"],
            ft1       = 'Variable1',
            ft2       = 'Variable2',
        ) 

        ## 2a. Significant Relationships between Categorical and Continuous Variables (Parametric)
        strong_findings_paths["categorical_continuous_parametric"] = filter_list(
            file_list = strong_findings_paths["categorical_continuous_parametric"],
            df        = strong_findings["categorical_continuous_parametric"],
            ft1       = 'Categorical',
            ft2       = 'Continuous',
        ) 

        ## 2b. Strong Relationships between Categorical and Continuous Variables (Non-parametric)
        strong_findings_paths["categorical_continuous_nonparametric"] = filter_list(
            file_list = strong_findings_paths["categorical_continuous_nonparametric"],
            df        = strong_findings["categorical_continuous_nonparametric"],
            ft1       = 'Categorical',
            ft2       = 'Continuous',
        ) 

        ## 3b. Significant Non-parametric Correlations between Continuous Variables:
        strong_findings_paths["continuous_continuous_nonparametric"] = filter_list(
            file_list = strong_findings_paths["continuous_continuous_nonparametric"],
            df        = strong_findings["continuous_continuous_nonparametric"],
            ft1       = 'Variable1',
            ft2       = 'Variable2',
        ) 
        return strong_findings_paths






    def get_all_filters(self, args) -> Dict[str, List[str]]:
        filters_str = {
            "categorical_categorical" : [
                f"DOF filter (>= {args.chi_square_dof_min})",
                f"Cramér's V filter (>= {args.chi_square_cramer_v_min})",
                f"Power filter (>= {args.chi_square_power_min})"
            ],

            "categorical_continuous_parametric" : [
                f"Power filter (>= {args.anova_power_min})",
                f"Effect Size Cohen's d (>= {args.anova_cohens_d_min})",
                f"Effect Size ε² (>= {args.anova_epsilon_squared_min})",
                f"Effect Size Partial η² (>= {args.anova_eta_squared_min})",
                f"Effect Size CLES (diff >= {args.anova_cles_diff_min})"
            ],

            "categorical_continuous_nonparametric" : [
                f"Power filter (>= {args.nonparam_power_min})",
                f"Effect Size ε² (>= {args.nonparam_epsilon_squared_min})",
                f"Effect Size CLES (diff >= {args.nonparam_cles_diff_min})"

            ],

            "continuous_continuous_parametric": [
                f"Correlation Strength filter (|r| >= {args.pearson_corr_min})",
                f"Power filter (>= {args.pearson_power_min})"
            ],

            "continuous_continuous_nonparametric": [
                f"Correlation Strength filter (|r| >= {args.spearman_corr_min})",
                f"Power filter (>= {args.spearman_power_min})"
            ]
        }

        return filters_str


    def generate_final_report(
            self, 
            dir_static             : str,
            questionnaire_name     : str,
            filters                : Dict[str, List[str]],
            chi2_results           : pd.DataFrame,
            anova_results          : pd.DataFrame,
            nonparam_group_results : pd.DataFrame,
            pearson_results        : pd.DataFrame,
            spearman_results        : pd.DataFrame,
            include_significant_findings : bool = False
        ) : 
        """ Generate a summary of significant  adn strong findings across all tests """


        # Generate a summary of significant findings across all tests
        significant_findings = {
            "categorical_categorical"              : chi2_results[chi2_results["Significant"] == True],
            "categorical_continuous_parametric"    : anova_results[anova_results["Significant"] == True],
            "categorical_continuous_nonparametric" : nonparam_group_results[nonparam_group_results["Significant"] == True],
            "continuous_continuous_parametric"     : pearson_results[pearson_results["Significant"] == True],
            "continuous_continuous_nonparametric"  : spearman_results[spearman_results["Significant"] == True]
        }
        
        # Generate a summary of findings that are both significant AND passed filter criteria
        strong_findings = {
            "categorical_categorical": chi2_results[
                (chi2_results["Significant"] == True) & 
                (chi2_results["Passed_All_Filters"] == True)
                #(chi2_results["Passed_Any_Filter"] == True if "Passed_Any_Filter" in chi2_results.columns else True)
            ],
            "categorical_continuous_parametric": anova_results[
                (anova_results["Significant"] == True) & 
                (anova_results["Passed_All_Filters"] == True)
            ],
            "categorical_continuous_nonparametric": nonparam_group_results[
                (nonparam_group_results["Significant"] == True) & 
                (nonparam_group_results["Passed_All_Filters"] == True)
            ],
            "continuous_continuous_parametric": pearson_results[
                (pearson_results["Significant"] == True) & 
                (pearson_results["Passed_All_Filters"] == True)
            ],
            "continuous_continuous_nonparametric": spearman_results[
                (spearman_results["Significant"] == True) & 
                (spearman_results["Passed_All_Filters"] == True)
            ]
        }

        # Save summary of findings
        with open(f"{dir_static}/{questionnaire_name}/significant_findings_summary.txt", "w") as f:
            if include_significant_findings :
                f.write(f"Statistical Analysis Summary for {questionnaire_name}\n")
                f.write("="*60 + "\n\n")
                
                f.write("1. Significant Relationships between Categorical Variables:\n")
                f.write("-"*60 + "\n")
                for fl in filters["categorical_categorical"] : 
                    f.write(f"{fl}\n")
                if len(significant_findings["categorical_categorical"]) > 0:
                    for _, row in significant_findings["categorical_categorical"].iterrows():
                        passed_filter = "✓" if "Passed_Any_Filter" in row and row["Passed_Any_Filter"] else ""
                        f.write(f"  * {row['Variable1']} and {row['Variable2']} ({row['Test']}, p={row['p_value']:.4f}) {passed_filter}\n")
                else:
                    f.write("  * No significant relationships found\n")

                
                f.write("\n2a. Significant Relationships between Categorical and Continuous Variables (Parametric):\n")
                f.write("-"*60 + "\n")
                for fl in filters["categorical_continuous_parametric"] : 
                    f.write(f"{fl}\n")
                if len(significant_findings["categorical_continuous_parametric"]) > 0:
                    for _, row in significant_findings["categorical_continuous_parametric"].iterrows():
                        passed_filter = "✓" if row["Passed_Any_Filter"] else ""
                        f.write(f"  * {row['Categorical']} affects {row['Continuous']} ({row['Test']}, p={row['p_value']:.4f}) {passed_filter}\n")
                else:
                    f.write("  * No significant relationships found\n")
                
                f.write("\n2b. Significant Relationships between Categorical and Continuous Variables (Non-parametric):\n")
                f.write("-"*60 + "\n")
                for fl in filters["categorical_continuous_nonparametric"] : 
                    f.write(f"{fl}\n")
                if len(significant_findings["categorical_continuous_nonparametric"]) > 0:
                    for _, row in significant_findings["categorical_continuous_nonparametric"].iterrows():
                        passed_filter = "✓" if row["Passed_Any_Filter"] else ""
                        f.write(f"  * {row['Categorical']} affects {row['Continuous']} ({row['Test']}, p={row['p_value']:.4f}) {passed_filter}\n")
                else:
                    f.write("  * No significant relationships found\n")
                
                f.write("\n3a. Significant Parametric Correlations between Continuous Variables:\n")
                f.write("-"*60 + "\n")
                for fl in filters["continuous_continuous_parametric"] : 
                    f.write(f"{fl}\n")
                if len(significant_findings["continuous_continuous_parametric"]) > 0:
                    for _, row in significant_findings["continuous_continuous_parametric"].iterrows():
                        passed_filter = "✓" if row["Passed_Any_Filter"] else ""
                        f.write(f"  * {row['Variable1']} and {row['Variable2']} (r={row['Correlation']:.4f}, p={row['p_value']:.4f}) {passed_filter}\n")
                else:
                    f.write("  * No significant parametric correlations found\n")
                
                f.write("\n3b. Significant Non-parametric Correlations between Continuous Variables:\n")
                f.write("-"*60 + "\n")
                for fl in filters["continuous_continuous_nonparametric"] : 
                    f.write(f"{fl}\n")
                if len(significant_findings["continuous_continuous_nonparametric"]) > 0:
                    for _, row in significant_findings["continuous_continuous_nonparametric"].iterrows():
                        passed_filter = "✓" if row["Passed_Any_Filter"] else ""
                        f.write(f"  * {row['Variable1']} and {row['Variable2']} (rho={row['Correlation']:.4f}, p={row['p_value']:.4f}) {passed_filter}\n")
                else:
                    f.write("  * No significant non-parametric correlations found\n")
            
            # Add section for strong findings (both significant and pass filter criteria)
            f.write("\n\nSTRONG FINDINGS (Significant + Passed Quality Filters)\n")
            f.write("="*60 + "\n\n")
            
            # Count total strong findings
            total_strong = sum(len(v) for v in strong_findings.values())
            if total_strong > 0:
                for category, findings in strong_findings.items():
                    if len(findings) > 0:
                        if "categorical_categorical" in category:
                            f.write("1. Strong Relationships between Categorical Variables:\n")
                            f.write("-"*60 + "\n")
                            for fl in filters["categorical_categorical"] : 
                                f.write(f"{fl}\n")
                            for _, row in findings.iterrows():
                                f.write(f"  * {row['Variable1']} and {row['Variable2']} ({row['Test']}, p={row['p_value']:.4f})\n")
                        
                        elif "categorical_continuous_parametric" in category:
                            f.write("\n2a. Significant Relationships between Categorical and Continuous Variables (Parametric):\n")
                            f.write("-"*60 + "\n")
                        
                            for fl in filters["categorical_continuous_parametric"] : 
                                f.write(f"{fl}\n")
                            for _, row in findings.iterrows():
                                f.write(f"  * {row['Categorical']} affects {row['Continuous']} ({row['Test']}, p={row['p_value']:.4f})\n")
                        
                        elif "categorical_continuous_nonparametric" in category:
                            f.write("\n2b. Strong Relationships between Categorical and Continuous Variables (Non-parametric):\n")
                            f.write("-"*60 + "\n")
                            for fl in filters["categorical_continuous_nonparametric"] : 
                                f.write(f"{fl}\n")
                            for _, row in findings.iterrows():
                                f.write(f"  * {row['Categorical']} affects {row['Continuous']} ({row['Test']}, p={row['p_value']:.4f})\n")
                        
                        elif "continuous_continuous_parametric" in category:
                            f.write("\n3a. Strong Parametric Correlations between Continuous Variables:\n")
                            f.write("-"*60 + "\n")
                            for fl in filters["continuous_continuous_parametric"] : 
                                f.write(f"{fl}\n")
                            for _, row in findings.iterrows():
                                f.write(f"  * {row['Variable1']} and {row['Variable2']} (r={row['Correlation']:.4f}, p={row['p_value']:.4f})\n")
                        
                        elif "continuous_continuous_nonparametric" in category:
                            f.write("\n3b. Significant Non-parametric Correlations between Continuous Variables:\n")
                            f.write("-"*60 + "\n")
                            for fl in filters["continuous_continuous_nonparametric"] : 
                                f.write(f"{fl}\n")
                            for _, row in findings.iterrows():
                                f.write(f"  * {row['Variable1']} and {row['Variable2']} (rho={row['Correlation']:.4f}, p={row['p_value']:.4f})\n")
            else:
                f.write("No findings met both significance and filter criteria.\n")
            
        print(f"Analysis complete. Results saved to {dir_static}/{questionnaire_name}/")




    
    def get_pairs_of_variables(
            self, 
            data               = None,
            categorial_exclude = ["Timestamp", "Start Time", "Email"]
        ):
        """
        Generate pairs of variables for statistical testing.
        
        Args:
            data: DataFrame containing the variables
        
        Returns:
            Dictionary of pairs categorized by test type
        """
        pairs = {
            "categorical_categorical": [],
            "categorical_continuous": [],
            "continuous_continuous": []
        }
        
        if data is not None:
            # First identify questionnaire items (Q1-Q10)
            questionnaire_items = [col for col in data.columns if col.startswith('Q') and col[1:].isdigit()]
            
            # Auto-detect variable types
            # Treat questionnaire items as continuous regardless of their cardinality
            categorical_vars = [col for col in data.columns if 
                               (col not in questionnaire_items) and  # Exclude questionnaire items
                               (data[col].dtype == 'object' or 
                               (data[col].dtype == 'int64' and data[col].nunique() < 10))]
            
            # Include questionnaire items as continuous variables
            continuous_vars = [col for col in data.columns if 
                              col in questionnaire_items or  # Include all questionnaire items
                              (col not in categorical_vars and data[col].dtype in ['int64', 'float64'])]
            
            # Exclude specific categorical 
            categorical_vars_after_exclusions = [v for v in categorical_vars if v not in categorial_exclude]
            
            print(f"Detected variable types:")
            print(f"  Categorical: {categorical_vars}")
            print(f"  Categorical (after exclusions): {categorical_vars_after_exclusions}")
            print(f"  Continuous (includes ordinal questionnaire items): {continuous_vars}")
            print(f"  Questionnaire items (treated as ordinal): {questionnaire_items}")
            
            # Categorical-Categorical pairs
            for i, var1 in enumerate(categorical_vars_after_exclusions):
                for var2 in categorical_vars_after_exclusions[i+1:]:
                    pairs["categorical_categorical"].append((var1, var2))
            
            # Categorical-Continuous pairs
            for cat_var in categorical_vars_after_exclusions:
                for cont_var in continuous_vars:
                    pairs["categorical_continuous"].append((cat_var, cont_var))
            
            # Continuous-Continuous pairs
            for i, var1 in enumerate(continuous_vars):
                for var2 in continuous_vars[i+1:]:
                    pairs["continuous_continuous"].append((var1, var2))
        
        return pairs
    
    def chi2_test_wrapper(self, data, pairs=None, directory=None):
        """
        Perform Chi-square test for categorical-categorical relationships using pingouin.
        
        Args:
            data: DataFrame containing the variables
            pairs: List of tuples containing pairs of categorical variables
            directory: Directory to save visualizations (if any)
        
        Returns:
            DataFrame containing test results, filtered by statistical criteria
        """
        if pairs is None:
            all_pairs = self.get_pairs_of_variables(data)
            pairs = all_pairs["categorical_categorical"]
        
        results_list = []
        
        for var1, var2 in tqdm(pairs, desc="Running Chi-square tests"):
            if var1 in data.columns and var2 in data.columns:
                try:
                    # Create contingency table
                    contingency = pd.crosstab(data[var1], data[var2])
                    
                    # Skip if not enough unique values
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        print(f"Skipping {var1} vs {var2}: Contingency table has less than 2 rows or columns")
                        continue
                    
                    # Skip if empty contingency table
                    if contingency.size == 0 or contingency.values.sum() == 0:
                        print(f"Skipping {var1} vs {var2}: Empty contingency table")
                        continue
                    
                    # Use pingouin's chi2_independence function
                    try:
                        # According to pingouin documentation, chi2_independence returns:
                        # expected, observed, stats
                        expected, observed, stats = pg.chi2_independence(data, x=var1, y=var2)
                        
                        # Stats is a DataFrame containing the test results
                        # It should contain columns: 'chi2', 'dof', 'pval', 'cramer'
                        chi2 = stats['chi2'].values[0]
                        p_value = stats['pval'].values[0]
                        dof = stats['dof'].values[0]
                        cramer_v = stats['cramer'].values[0]
                        test_used = "Chi-square"
                        statistic = chi2
                        
                        # Calculate statistical power
                        # Power calculation for chi-square (approximate)
                        n = contingency.values.sum()
                        effect_size = cramer_v
                        power = pg.power_chi2(dof=(dof), w=effect_size, n=n, alpha=self.significance_threshold)
                        
                        # Check if any expected cell count is less than 5
                        low_expected = (expected < 5).any().any()
                        if low_expected:
                            print(f"Warning: Low expected frequencies for {var1} vs {var2}, consider Fisher's exact test")
                    
                    except Exception as e:
                        print(f"Error using pingouin.chi2_independence for {var1} vs {var2}: {str(e)}")
                        print("Falling back to scipy.stats.chi2_contingency")
                        
                        # Fallback to scipy
                        from scipy import stats as scipy_stats
                        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
                        
                        # Calculate Cramer's V manually
                        n = contingency.values.sum()
                        min_dim = min(contingency.shape) - 1
                        cramer_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else np.nan
                        
                        # Calculate power (approximate)
                        power = pg.power_chi2(dof=(dof), w=cramer_v, n=n, alpha=self.significance_threshold)
                        
                        test_used = "Chi-square (scipy)"
                        statistic = chi2
                        low_expected = False  # We don't have easy access to this with scipy
                    
                    # For 2x2 tables, consider using Fisher's exact test
                    if contingency.shape == (2, 2):
                        try:
                            # For 2x2 tables, use pingouin's fisher_exact
                            result = pg.fisher_exact(contingency)
                            
                            # Extract results - fisher_exact returns a DataFrame with:
                            # columns: 'oddsratio', 'p-val', 'CI95%'
                            odds_ratio = result['oddsratio'].values[0]
                            fisher_p = result['p-val'].values[0]
                            
                            # Calculate power for Fisher's exact test (approximate)
                            n = contingency.values.sum()
                            # Use phi coefficient as effect size for 2x2 tables
                            phi = cramer_v  # For 2x2 tables, phi = Cramer's V
                            fisher_power = pg.power_chi2(dof=1, w=phi, n=n, alpha=self.significance_threshold)
                            
                            # Add Fisher's results to the list
                            fisher_significant = fisher_p < self.significance_threshold
                            
                            results_list.append({
                                "Variable1": var1,
                                "Variable2": var2,
                                "Test": "Fisher's Exact",
                                "Statistic": odds_ratio,
                                "p_value": fisher_p,
                                "Significant": fisher_significant,
                                "DOF": 1,  # Fixed for 2x2 tables
                                "Cramer_V": phi,
                                "Power": fisher_power,
                                "Low_Expected": low_expected
                            })
                            
                        except Exception as fisher_err:
                            print(f"Fisher's exact test failed for {var1} vs {var2}: {str(fisher_err)}")
                    
                    # Add Chi-square result to list
                    chi_significant = p_value < self.significance_threshold
                    
                    results_list.append({
                        "Variable1": var1,
                        "Variable2": var2,
                        "Test": test_used,
                        "Statistic": statistic,
                        "p_value": p_value,
                        "Significant": chi_significant,
                        "DOF": dof,
                        "Cramer_V": cramer_v,
                        "Power": power,
                        "Low_Expected": low_expected
                    })
                    
                    # Create visualization if directory provided
                    if directory and (chi_significant or (contingency.shape == (2, 2) and fisher_significant)):
                        try:
                            # Create visualization with pandas and matplotlib
                            plt.figure(figsize=(10, 6))
                            sns.heatmap(contingency, annot=True, cmap='YlGnBu', fmt='d')
                            plt.title(f'Contingency Table: {var1} vs {var2}')
                            plt.tight_layout()
                            plt.savefig(f"{directory}/{var1}_{var2}_contingency.png")
                            plt.close()
                            
                            # Create normalized heatmap
                            plt.figure(figsize=(10, 6))
                            sns.heatmap(pd.crosstab(data[var1], data[var2], normalize='index'), 
                                      annot=True, cmap='YlGnBu', fmt='.2%')
                            plt.title(f'Relationship between {var1} and {var2}')
                            plt.tight_layout()
                            plt.savefig(f"{directory}/{var1}_{var2}_heatmap.png")
                            plt.close()
                            
                            # Optional: Plot expected vs observed counts
                            if isinstance(expected, pd.DataFrame) and isinstance(observed, pd.DataFrame):
                                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                                sns.heatmap(observed, annot=True, cmap='Blues', fmt='d', ax=axes[0])
                                axes[0].set_title(f'Observed Counts: {var1} vs {var2}')
                                
                                sns.heatmap(expected, annot=True, cmap='Oranges', fmt='.1f', ax=axes[1])
                                axes[1].set_title(f'Expected Counts: {var1} vs {var2}')
                                
                                plt.tight_layout()
                                plt.savefig(f"{directory}/{var1}_{var2}_observed_expected.png")
                                plt.close()
                                
                        except Exception as viz_err:
                            print(f"Error creating visualization for {var1} vs {var2}: {str(viz_err)}")
                        
                except Exception as e:
                    print(f"Error testing {var1} vs {var2}: {str(e)}")
                    continue
        
        # Convert results to DataFrame
        if not results_list:
            print("Warning: No valid test results were generated.")
            return pd.DataFrame(columns=["Variable1", "Variable2", "Test", "Statistic", "p_value", 
                                         "Significant", "DOF", "Cramer_V", "Power", "Low_Expected"])
        
        results_df = pd.DataFrame(results_list)
        
        # Apply filtering criteria based on rules of thumb
        # Store the original results for reference
        original_results = results_df.copy()
        
        # Filter based on DOF
        min_dof = 1  # Adjust this based on your specific needs (5 might be too stringent)
        dof_filtered = results_df[results_df['DOF'] >= min_dof]
        
        # Filter based on Cramér's V (association strength)
        min_cramer_v = 0.1  # Weak association threshold
        cramer_filtered = dof_filtered[dof_filtered['Cramer_V'] >= min_cramer_v]
        
        # Filter based on Power
        min_power = 0.8  # Conventional threshold for adequate power
        power_filtered = cramer_filtered[cramer_filtered['Power'] >= min_power]
        
        # Print filtering results
        print(f"\nFiltering Summary for Chi-square/Fisher's Tests:")
        print(f"Original results: {len(results_df)} tests")
        print(f"After DOF filter (>= {min_dof}): {len(dof_filtered)} tests")
        print(f"After Cramér's V filter (>= {min_cramer_v}): {len(cramer_filtered)} tests")
        print(f"After Power filter (>= {min_power}): {len(power_filtered)} tests")
        
        # If all filtering leaves no results, return the original with a warning
        if len(power_filtered) == 0:
            print("Warning: Filtering criteria removed all results. Returning original results.")
            # Add a column indicating which results passed all filters
            original_results['Passed_All_Filters'] = False
            return original_results
        
        # Add a column indicating which results passed all filters
        results_df['Passed_All_Filters'] = False
        for idx in power_filtered.index:
            results_df.loc[idx, 'Passed_All_Filters'] = True
        
        return results_df
    
    
    
    
    
    def nonparametric_group_test_wrapper(
            self, 
            data, 
            epsilon_squared_min = 0.02,
            cles_diff_min       = 0.05,
            power_min           = 0.5,
            pairs               = None, 
            directory           = None
        ):
        """
        Perform nonparametric tests for categorical-continuous relationships using pingouin.
        
        Args:
            data: DataFrame containing the variables
            pairs: List of tuples containing pairs of (categorical, continuous) variables
            directory: Directory to save visualizations (if any)
        
        Returns:
            DataFrame containing test results
        """
        if pairs is None:
            all_pairs = self.get_pairs_of_variables(data)
            pairs = all_pairs["categorical_continuous"]
        
        results_list = []
        
        for cat_var, cont_var in tqdm(pairs, desc="Running nonparametric group tests"):
            if cat_var in data.columns and cont_var in data.columns:
                try:
                    # Create a filtered dataframe with no NaN values
                    filtered_data = data[[cat_var, cont_var]].dropna()
                    
                    # Skip if not enough data
                    if len(filtered_data) < 3:
                        print(f"Skipping {cat_var} vs {cont_var}: Not enough data")
                        continue
                    
                    # Get unique categories
                    categories = filtered_data[cat_var].unique()
                    
                    if len(categories) < 2:
                        print(f"Skipping {cat_var} vs {cont_var}: Need at least 2 categories")
                        continue
                    
                    # Prepare groups for analysis
                    groups = []
                    group_sizes = []
                    group_data_by_category = {}
                    
                    for category in categories:
                        group_data = filtered_data[filtered_data[cat_var] == category][cont_var]
                        if len(group_data) >= 2:  # Need at least 2 samples per group
                            groups.append(group_data.values)
                            group_sizes.append(len(group_data))
                            group_data_by_category[category] = group_data.values
                    
                    if len(groups) < 2:
                        print(f"Skipping {cat_var} vs {cont_var}: Not enough valid groups")
                        continue
                    
                    # Choose appropriate test based on number of categories
                    if len(groups) == 2:
                        # Two groups: Mann-Whitney U test
                        try:
                            group_values = list(group_data_by_category.values())
                            mw_result = pg.mwu(group_values[0], group_values[1])
                            test_used = "Mann-Whitney U"
                            statistic = mw_result['U-val'].values[0]
                            p_value = mw_result['p-val'].values[0]
                            
                            # Calculate effect size (CLES - Common Language Effect Size)
                            effect_size = mw_result['CLES'].values[0]
                            effect_type = "CLES"
                            
                            # Calculate rank-biserial correlation (another effect size)
                            n1, n2 = len(group_values[0]), len(group_values[1])
                            n_total = n1 + n2
                            rank_biserial = 1 - (2 * statistic) / (n1 * n2)
                            
                            # Approximate power calculation for Mann-Whitney U test
                            # Convert to equivalent Cohen's d for estimation
                            means = [np.mean(g) for g in group_values]
                            pooled_std = np.sqrt(np.sum([(len(g) - 1) * np.var(g) for g in group_values]) / (n_total - 2))
                            equiv_d = abs(means[0] - means[1]) / pooled_std if pooled_std > 0 else 0
                            
                            # Adjust for non-parametric efficiency (~0.95 for large samples)
                            equiv_d_adjusted = equiv_d * 0.95  
                            power = pg.power_ttest2n(d=equiv_d_adjusted, nx=n1, ny=n2, alpha=self.significance_threshold)
                            
                        except Exception as e:
                            print(f"Error in Mann-Whitney U test for {cat_var} vs {cont_var}: {str(e)}")
                            continue
                            
                    else:
                        # More than two groups: Kruskal-Wallis test
                        try:
                            kw_result = pg.kruskal(dv=cont_var, between=cat_var, data=filtered_data)
                            test_used = "Kruskal-Wallis"
                            statistic = kw_result['H'].values[0]
                            p_value = kw_result['p-unc'].values[0]
                            
                            # Calculate effect size (epsilon-squared)
                            n = filtered_data.shape[0]
                            k = len(groups)
                            dof = k - 1
                            epsilon_squared = (statistic - dof) / (n - k)
                            effect_size = max(0, epsilon_squared)  # Ensure non-negative
                            effect_type = "ε²"
                            rank_biserial = np.nan  # Not applicable for >2 groups
                            
                            # Approximate power calculation for Kruskal-Wallis
                            # Convert epsilon-squared to Cohen's f for power approximation
                            equiv_f = np.sqrt(effect_size / (1 - effect_size)) if effect_size < 1 else 1.0
                            power = pg.power_anova(eta_squared=effect_size, k=k, n=n, alpha=self.significance_threshold)
                            
                        except Exception as e:
                            print(f"Error in Kruskal-Wallis test for {cat_var} vs {cont_var}: {str(e)}")
                            continue
                    
                    # Check if result is significant
                    significant = p_value < self.significance_threshold
                    
                    # Add result to list
                    results_list.append({
                        "Categorical": cat_var,
                        "Continuous": cont_var,
                        "Test": test_used,
                        "Statistic": statistic,
                        "p_value": p_value,
                        "Significant": significant,  # Explicitly include the Significant column
                        "Groups": len(groups),
                        "Total_n": sum(group_sizes),
                        "Effect_Size": effect_size,
                        "Effect_Type": effect_type,
                        "Rank_Biserial": rank_biserial if 'rank_biserial' in locals() else np.nan,
                        "Power": power
                    })
                    
                    # Perform post-hoc tests if significant and more than 2 groups
                    if significant and len(groups) > 2:
                        try:
                            # Non-parametric post-hoc: Dunn's test
                            posthoc = pg.pairwise_tests(dv=cont_var, between=cat_var, data=filtered_data, 
                                                        parametric=False, padjust='bonf')
                            
                            # Save post-hoc results
                            if directory:
                                posthoc.to_csv(f"{directory}/{cat_var}_{cont_var}_posthoc_dunn.csv", index=False)
                                
                        except Exception as posthoc_err:
                            print(f"Error in post-hoc test for {cat_var} vs {cont_var}: {str(posthoc_err)}")
                    
                    # Create visualization if directory provided
                    if directory:
                        try:
                            # Always create visualizations regardless of significance
                            plt.figure(figsize=(12, 6))
                            sns.boxplot(x=cat_var, y=cont_var, data=filtered_data)
                            title = f'Distribution of {cont_var} by {cat_var}\n({test_used}, p={p_value:.4f})'
                            if significant:
                                title += " *"
                            plt.title(title)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(f"{directory}/{cat_var}_{cont_var}_nonparametric_boxplot.png")
                            plt.close()
                            
                            # Create violin plot for better visualization of distributions
                            plt.figure(figsize=(12, 6))
                            sns.violinplot(x=cat_var, y=cont_var, data=filtered_data, inner="points")
                            plt.title(title)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(f"{directory}/{cat_var}_{cont_var}_nonparametric_violin.png")
                            plt.close()
                            
                        except Exception as viz_err:
                            print(f"Error creating visualization for {cat_var} vs {cont_var}: {str(viz_err)}")
                    
                except Exception as e:
                    print(f"Error testing {cat_var} vs {cont_var}: {str(e)}")
                    continue
        
        # Convert results to DataFrame
        if not results_list:
            print("Warning: No valid test results were generated.")
            return pd.DataFrame(columns=["Categorical", "Continuous", "Test", "Statistic", "p_value", 
                                         "Significant", "Groups", "Total_n", "Effect_Size", "Effect_Type", 
                                         "Rank_Biserial", "Power"])
        
        results_df = pd.DataFrame(results_list)
        
        # Apply filtering criteria based on rules of thumb - with MORE LENIENT thresholds
        # Store the original results for reference
        original_results = results_df.copy()
        
        # Filter based on Effect Size - REDUCED THRESHOLDS
        effect_size_filtered = results_df.copy()
        for idx, row in results_df.iterrows():
            keep_row = True
            if row['Effect_Type'] == "CLES":
                if abs(row['Effect_Size'] - 0.5) < cles_diff_min:  # CLES should differ from 0.5 by at least 0.05 (reduced from 0.1)
                    keep_row = False
            elif row['Effect_Type'] == "ε²":
                if row['Effect_Size'] < epsilon_squared_min:  # Small effect (reduced from 0.06)
                    keep_row = False
        
            if not keep_row:
                effect_size_filtered = effect_size_filtered.drop(idx)
        
        # Filter based on Power - REDUCED THRESHOLD
        power_filtered = results_df[results_df['Power'] >= power_min]
        
        # Combine filters with OR instead of AND to be more inclusive
        combined_filtered = pd.concat([effect_size_filtered, power_filtered]).drop_duplicates()

        # Combine filters with AND 
        all_filtered = pd.concat([effect_size_filtered, power_filtered], axis=1, join="inner").dropna()
        
        # Print filtering results
        print(f"\nFiltering Summary for Nonparametric Group Tests:")
        print(f"Original results: {len(results_df)} tests")
        print(f"After Effect Size filter (small or larger): {len(effect_size_filtered)} tests")
        print(f"After Power filter (>= {power_min}): {len(power_filtered)} tests")
        print(f"After all filters (strict): {len(all_filtered)} tests")
        print(f"After combined filters (less strict): {len(combined_filtered)} tests")
        
        # Include ALL results in the output DataFrame
        results_df['Passed_Effect_Size_Filter'] = False
        results_df['Passed_Power_Filter'] = False
        results_df['Passed_Any_Filter'] = False
        results_df['Passed_All_Filters'] = False
        
        for idx in results_df.index:
            # Check if it passes the effect size filter
            passes_effect_size = True
            if results_df.loc[idx, 'Effect_Type'] == "CLES":
                if abs(results_df.loc[idx, 'Effect_Size'] - 0.5) < cles_diff_min:
                    passes_effect_size = False
            elif results_df.loc[idx, 'Effect_Type'] == "ε²":
                if results_df.loc[idx, 'Effect_Size'] < epsilon_squared_min:
                    passes_effect_size = False
                
            if passes_effect_size:
                results_df.loc[idx, 'Passed_Effect_Size_Filter'] = True
                
            # Check if it passes the power filter
            if results_df.loc[idx, 'Power'] >= power_min:
                results_df.loc[idx, 'Passed_Power_Filter'] = True
                
            # Check if it passes any filter
            if idx in combined_filtered.index:
                results_df.loc[idx, 'Passed_Any_Filter'] = True

            # Check if it passes any filter
            if idx in all_filtered.index:
                results_df.loc[idx, 'Passed_All_Filters'] = True
        
        # IMPORTANT: Return ALL results, not just filtered ones
        return results_df
    





    
    def chi2_test_wrapper(
            self, 
            data, 
            dof_min      = 1,
            cramer_v_min = 0.1,
            power_min    = 0.8,
            pairs        = None, 
            directory    = None,
        ):
        """
        Perform Chi-square test for categorical-categorical relationships using pingouin.
        
        Args:
            data: DataFrame containing the variables
            pairs: List of tuples containing pairs of categorical variables
            directory: Directory to save visualizations (if any)
        
        Returns:
            DataFrame containing test results, filtered by statistical criteria
        """
        if pairs is None:
            all_pairs = self.get_pairs_of_variables(data)
            pairs = all_pairs["categorical_categorical"]
        
        results_list = []
        
        for var1, var2 in tqdm(pairs, desc="Running Chi-square tests"):
            if var1 in data.columns and var2 in data.columns:
                try:
                    # Create contingency table
                    contingency = pd.crosstab(data[var1], data[var2])
                    
                    # Skip if not enough unique values
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        print(f"Skipping {var1} vs {var2}: Contingency table has less than 2 rows or columns")
                        continue
                    
                    # Skip if empty contingency table
                    if contingency.size == 0 or contingency.values.sum() == 0:
                        print(f"Skipping {var1} vs {var2}: Empty contingency table")
                        continue
                    
                    # Use pingouin's chi2_independence function
                    try:
                        # According to pingouin documentation, chi2_independence returns:
                        # expected, observed, stats
                        expected, observed, stats = pg.chi2_independence(data, x=var1, y=var2)
                        
                        # Stats is a DataFrame containing the test results
                        # It should contain columns: 'chi2', 'dof', 'pval', 'cramer'
                        chi2 = stats['chi2'].values[0]
                        p_value = stats['pval'].values[0]
                        dof = stats['dof'].values[0]
                        cramer_v = stats['cramer'].values[0]
                        test_used = "Chi-square"
                        statistic = chi2
                        
                        # Calculate statistical power
                        # Power calculation for chi-square (approximate)
                        n = contingency.values.sum()
                        effect_size = cramer_v
                        power = pg.power_chi2(dof=(dof), w=effect_size, n=n, alpha=self.significance_threshold)
                        
                        # Check if any expected cell count is less than 5
                        low_expected = (expected < 5).any().any()
                        if low_expected:
                            print(f"Warning: Low expected frequencies for {var1} vs {var2}, consider Fisher's exact test")
                    
                    except Exception as e:
                        print(f"Error using pingouin.chi2_independence for {var1} vs {var2}: {str(e)}")
                        print("Falling back to scipy.stats.chi2_contingency")
                        
                        # Fallback to scipy
                        from scipy import stats as scipy_stats
                        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
                        
                        # Calculate Cramer's V manually
                        n = contingency.values.sum()
                        min_dim = min(contingency.shape) - 1
                        cramer_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else np.nan
                        
                        # Calculate power (approximate)
                        power = pg.power_chi2(dof=(dof), w=cramer_v, n=n, alpha=self.significance_threshold)
                        
                        test_used = "Chi-square (scipy)"
                        statistic = chi2
                        low_expected = False  # We don't have easy access to this with scipy
                    
                    # For 2x2 tables, consider using Fisher's exact test
                    if contingency.shape == (2, 2):
                        try:
                            # For 2x2 tables, use pingouin's fisher_exact
                            result = pg.fisher_exact(contingency)
                            
                            # Extract results - fisher_exact returns a DataFrame with:
                            # columns: 'oddsratio', 'p-val', 'CI95%'
                            odds_ratio = result['oddsratio'].values[0]
                            fisher_p = result['p-val'].values[0]
                            
                            # Calculate power for Fisher's exact test (approximate)
                            n = contingency.values.sum()
                            # Use phi coefficient as effect size for 2x2 tables
                            phi = cramer_v  # For 2x2 tables, phi = Cramer's V
                            fisher_power = pg.power_chi2(dof=1, w=phi, n=n, alpha=self.significance_threshold)
                            
                            # Add Fisher's results to the list
                            fisher_significant = fisher_p < self.significance_threshold
                            
                            results_list.append({
                                "Variable1": var1,
                                "Variable2": var2,
                                "Test": "Fisher's Exact",
                                "Statistic": odds_ratio,
                                "p_value": fisher_p,
                                "Significant": fisher_significant,
                                "DOF": 1,  # Fixed for 2x2 tables
                                "Cramer_V": phi,
                                "Power": fisher_power,
                                "Low_Expected": low_expected
                            })
                            
                        except Exception as fisher_err:
                            print(f"Fisher's exact test failed for {var1} vs {var2}: {str(fisher_err)}")
                    
                    # Add Chi-square result to list
                    chi_significant = p_value < self.significance_threshold
                    
                    results_list.append({
                        "Variable1": var1,
                        "Variable2": var2,
                        "Test": test_used,
                        "Statistic": statistic,
                        "p_value": p_value,
                        "Significant": chi_significant,
                        "DOF": dof,
                        "Cramer_V": cramer_v,
                        "Power": power,
                        "Low_Expected": low_expected
                    })
                    
                    # Create visualization if directory provided
                    if directory and (chi_significant or (contingency.shape == (2, 2) and fisher_significant)):
                        try:
                            # Create visualization with pandas and matplotlib
                            plt.figure(figsize=(10, 6))
                            sns.heatmap(contingency, annot=True, cmap='YlGnBu', fmt='d')
                            plt.title(f'Contingency Table: {var1} vs {var2}')
                            plt.tight_layout()
                            plt.savefig(f"{directory}/{var1}_{var2}_contingency.png")
                            plt.close()
                            
                            # Create normalized heatmap
                            plt.figure(figsize=(10, 6))
                            sns.heatmap(pd.crosstab(data[var1], data[var2], normalize='index'), 
                                      annot=True, cmap='YlGnBu', fmt='.2%')
                            plt.title(f'Relationship between {var1} and {var2}')
                            plt.tight_layout()
                            plt.savefig(f"{directory}/{var1}_{var2}_heatmap.png")
                            plt.close()
                            
                            # Optional: Plot expected vs observed counts
                            if isinstance(expected, pd.DataFrame) and isinstance(observed, pd.DataFrame):
                                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                                sns.heatmap(observed, annot=True, cmap='Blues', fmt='d', ax=axes[0])
                                axes[0].set_title(f'Observed Counts: {var1} vs {var2}')
                                
                                sns.heatmap(expected, annot=True, cmap='Oranges', fmt='.1f', ax=axes[1])
                                axes[1].set_title(f'Expected Counts: {var1} vs {var2}')
                                
                                plt.tight_layout()
                                plt.savefig(f"{directory}/{var1}_{var2}_observed_expected.png")
                                plt.close()
                                
                        except Exception as viz_err:
                            print(f"Error creating visualization for {var1} vs {var2}: {str(viz_err)}")
                        
                except Exception as e:
                    print(f"Error testing {var1} vs {var2}: {str(e)}")
                    continue
        
        # Convert results to DataFrame
        if not results_list:
            print("Warning: No valid test results were generated.")
            return pd.DataFrame(columns=["Variable1", "Variable2", "Test", "Statistic", "p_value", 
                                         "Significant", "DOF", "Cramer_V", "Power", "Low_Expected"])
        
        results_df = pd.DataFrame(results_list)
        
        # Apply filtering criteria based on rules of thumb
        # Store the original results for reference
        original_results = results_df.copy()
        
        # Filter based on DOF
        dof_filtered = results_df[results_df['DOF'] >= dof_min]
        
        # Filter based on Cramér's V (association strength)
        cramer_filtered = dof_filtered[dof_filtered['Cramer_V'] >= cramer_v_min]
        
        # Filter based on Power
        power_filtered = cramer_filtered[cramer_filtered['Power'] >= power_min]
        
        # Print filtering results
        print(f"\nFiltering Summary for Chi-square/Fisher's Tests:")
        print(f"Original results: {len(results_df)} tests")
        print(f"After DOF filter (>= {dof_min}): {len(dof_filtered)} tests")
        print(f"After Cramér's V filter (>= {cramer_v_min}): {len(cramer_filtered)} tests")
        print(f"After Power filter (>= {power_min}): {len(power_filtered)} tests")
        
        # If all filtering leaves no results, return the original with a warning
        if len(power_filtered) == 0:
            print("Warning: Filtering criteria removed all results. Returning original results.")
            # Add a column indicating which results passed all filters
            original_results['Passed_All_Filters'] = False
            return original_results
        
        # Add a column indicating which results passed all filters
        results_df['Passed_All_Filters'] = False
        for idx in power_filtered.index:
            results_df.loc[idx, 'Passed_All_Filters'] = True
        
        return results_df
    


    
    def anova_test_wrapper(
            self, 
            data, 
            power_min           = 0.5,
            cohens_d_min        = 0.3,
            epsilon_squared_min = 0.03,
            eta_squared_min     = 0.03,
            cles_diff_min       = 0.1,
            pairs               = None, 
            directory           = None,
        ):
        """
        Perform ANOVA or t-test for categorical-continuous relationships using pingouin.
        
        Args:
            data: DataFrame containing the variables
            pairs: List of tuples containing pairs of (categorical, continuous) variables
            directory: Directory to save visualizations (if any)
        
        Returns:
            DataFrame containing test results
        """
        if pairs is None:
            all_pairs = self.get_pairs_of_variables(data)
            pairs = all_pairs["categorical_continuous"]
        
        results_list = []
        
        for cat_var, cont_var in tqdm(pairs, desc="Running ANOVA/t-tests"):
            if cat_var in data.columns and cont_var in data.columns:
                try:
                    # Create a filtered dataframe with no NaN values
                    filtered_data = data[[cat_var, cont_var]].dropna()
                    
                    # Skip if not enough data
                    if len(filtered_data) < 3:
                        print(f"Skipping {cat_var} vs {cont_var}: Not enough data")
                        continue
                    
                    # Get unique categories
                    categories = filtered_data[cat_var].unique()
                    
                    if len(categories) < 2:
                        print(f"Skipping {cat_var} vs {cont_var}: Need at least 2 categories")
                        continue
                    
                    # Prepare groups for analysis
                    groups = []
                    group_sizes = []
                    for category in categories:
                        group_data = filtered_data[filtered_data[cat_var] == category][cont_var]
                        if len(group_data) >= 2:  # Need at least 2 samples per group
                            groups.append(group_data.values)
                            group_sizes.append(len(group_data))
                    
                    if len(groups) < 2:
                        print(f"Skipping {cat_var} vs {cont_var}: Not enough valid groups")
                        continue
                    
                    # Check normality assumption for each group
                    normality_results = []
                    for i, group in enumerate(groups):
                        if len(group) >= 3:  # Need at least 3 samples for normality test
                            try:
                                # Use pingouin for normality test
                                norm_test = pg.normality(group)
                                normality_results.append(norm_test['normal'].values[0])
                            except Exception as e:
                                print(f"Error in normality test for {cat_var}={categories[i]}: {str(e)}")
                                normality_results.append(False)
                        else:
                            normality_results.append(False)
                    
                    normality_assumption = all(normality_results)
                    
                    # Test for homogeneity of variance
                    try:
                        # Use pingouin for homogeneity test
                        equal_var_test = pg.homoscedasticity(groups)
                        equal_var = equal_var_test['equal_var'].values[0]
                    except Exception as e:
                        print(f"Error in homogeneity test for {cat_var} vs {cont_var}: {str(e)}")
                        equal_var = False
                    
                    # Choose appropriate test based on conditions
                    if len(groups) == 2:
                        # Two groups: t-test or Mann-Whitney
                        if normality_assumption and equal_var:
                            # Parametric: Independent t-test
                            try:
                                ttest_result = pg.ttest(groups[0], groups[1], paired=False)
                                test_used = "t-test (equal var)"
                                statistic = ttest_result['T'].values[0]
                                p_value = ttest_result['p-val'].values[0]
                                dof = ttest_result['dof'].values[0]
                                cohens_d = ttest_result['cohen-d'].values[0]
                                power = ttest_result['power'].values[0]
                                effect_type = "Cohen's d"
                                effect_size = cohens_d
                            except Exception as e:
                                print(f"Error in t-test for {cat_var} vs {cont_var}: {str(e)}")
                                continue
                                
                        elif normality_assumption and not equal_var:
                            # Parametric: Welch's t-test
                            try:
                                ttest_result = pg.ttest(groups[0], groups[1], paired=False, correction=True)
                                test_used = "Welch's t-test"
                                statistic = ttest_result['T'].values[0]
                                p_value = ttest_result['p-val'].values[0]
                                dof = ttest_result['dof'].values[0]
                                cohens_d = ttest_result['cohen-d'].values[0]
                                power = ttest_result['power'].values[0]
                                effect_type = "Cohen's d"
                                effect_size = cohens_d
                            except Exception as e:
                                print(f"Error in Welch's t-test for {cat_var} vs {cont_var}: {str(e)}")
                                continue
                                
                        else:
                            # Non-parametric: Mann-Whitney U test
                            try:
                                mw_result = pg.mwu(groups[0], groups[1])
                                test_used = "Mann-Whitney U"
                                statistic = mw_result['U-val'].values[0]
                                p_value = mw_result['p-val'].values[0]
                                dof = "N/A"
                                
                                # Calculate effect size (common language effect size)
                                effect_size = mw_result['CLES'].values[0]
                                effect_type = "CLES"
                                
                                # Approximate power calculation
                                n1, n2 = len(groups[0]), len(groups[1])
                                # Use Cohen's d formula for rough power approximation
                                d = (np.mean(groups[0]) - np.mean(groups[1])) / np.sqrt((np.var(groups[0]) + np.var(groups[1])) / 2)
                                power = pg.power_ttest2n(d=abs(d), nx=n1, ny=n2, alpha=self.significance_threshold)
                            except Exception as e:
                                print(f"Error in Mann-Whitney U test for {cat_var} vs {cont_var}: {str(e)}")
                                continue
                    
                    else:
                        # More than two groups
                        if normality_assumption and equal_var:
                            # Parametric: One-way ANOVA
                            try:
                                anova_result = pg.anova(dv=cont_var, between=cat_var, data=filtered_data, detailed=True)
                                test_used = "One-way ANOVA"
                                statistic = anova_result['F'].values[0]
                                p_value = anova_result['p-unc'].values[0]
                                dof = f"{anova_result['ddof1'].values[0]}, {anova_result['ddof2'].values[0]}"
                                
                                # Calculate effect size (eta-squared)
                                effect_size = anova_result['np2'].values[0]  # Partial eta-squared
                                effect_type = "Partial η²"
                                
                                # Calculate power
                                f = np.sqrt(effect_size / (1 - effect_size))  # Convert eta-squared to Cohen's f
                                power = pg.power_anova(eta_squared=effect_size, k=len(groups), n=sum(group_sizes), alpha=self.significance_threshold)
                            except Exception as e:
                                print(f"Error in ANOVA for {cat_var} vs {cont_var}: {str(e)}")
                                continue
                        
                        else:
                            # Non-parametric: Kruskal-Wallis test
                            try:
                                kw_result = pg.kruskal(dv=cont_var, between=cat_var, data=filtered_data)
                                test_used = "Kruskal-Wallis"
                                statistic = kw_result['H'].values[0]
                                p_value = kw_result['p-unc'].values[0]
                                dof = kw_result['ddof1'].values[0]
                                
                                # Calculate effect size (epsilon-squared)
                                n = filtered_data.shape[0]
                                epsilon_squared = (statistic - dof + 1) / (n - dof)
                                effect_size = max(0, epsilon_squared)  # Ensure non-negative
                                effect_type = "ε²"
                                
                                # Approximate power calculation
                                # Convert epsilon-squared to Cohen's f for power approximation
                                f = np.sqrt(effect_size / (1 - effect_size)) if effect_size < 1 else 1.0
                                power = pg.power_anova(eta_squared=effect_size, k=len(groups), n=sum(group_sizes), alpha=self.significance_threshold)
                            except Exception as e:
                                print(f"Error in Kruskal-Wallis test for {cat_var} vs {cont_var}: {str(e)}")
                                continue
                    
                    # Check if result is significant
                    significant = p_value < self.significance_threshold
                    
                    # Add result to list with default filter flags (will be updated later)
                    results_list.append({
                        "Categorical": cat_var,
                        "Continuous": cont_var,
                        "Test": test_used,
                        "Statistic": statistic,
                        "p_value": p_value,
                        "Significant": significant,
                        "Groups": len(groups),
                        "DOF": dof,
                        "Effect_Size": effect_size,
                        "Effect_Type": effect_type,
                        "Power": power,
                        "Normality": normality_assumption,
                        "Equal_Variance": equal_var,
                        "Passed_Effect_Size_Filter": False,  # Will update later
                        "Passed_Power_Filter": False,        # Will update later
                        "Passed_Any_Filter": False           # Will update later
                    })
                    
                    # Perform post-hoc tests if significant and more than 2 groups
                    if significant and len(groups) > 2:
                        try:
                            # Choose appropriate post-hoc test
                            if normality_assumption and equal_var:
                                # Parametric: Tukey HSD
                                posthoc = pg.pairwise_tukey(dv=cont_var, between=cat_var, data=filtered_data)
                                posthoc_type = "Tukey HSD"
                            else:
                                # Non-parametric: Dunn's test
                                posthoc = pg.pairwise_tests(dv=cont_var, between=cat_var, data=filtered_data, parametric=False, padjust='bonf')
                                posthoc_type = "Dunn's Test"
                            
                            # Save post-hoc results
                            if directory:
                                posthoc.to_csv(f"{directory}/{cat_var}_{cont_var}_posthoc_{posthoc_type}.csv", index=False)
                                
                        except Exception as posthoc_err:
                            print(f"Error in post-hoc test for {cat_var} vs {cont_var}: {str(posthoc_err)}")
                    
                    # Create visualization if directory provided
                    if directory:
                        try:
                            # Always create visualizations regardless of significance
                            plt.figure(figsize=(12, 6))
                            sns.boxplot(x=cat_var, y=cont_var, data=filtered_data)
                            title = f'Distribution of {cont_var} by {cat_var}\n({test_used}, p={p_value:.4f})'
                            if significant:
                                title += " *"
                            plt.title(title)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(f"{directory}/{cat_var}_{cont_var}_boxplot.png")
                            plt.close()
                            
                            # Create violin plot for better distribution visualization
                            plt.figure(figsize=(12, 6))
                            sns.violinplot(x=cat_var, y=cont_var, data=filtered_data, inner="points")
                            plt.title(title)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(f"{directory}/{cat_var}_{cont_var}_violin.png")
                            plt.close()
                            
                        except Exception as viz_err:
                            print(f"Error creating visualization for {cat_var} vs {cont_var}: {str(viz_err)}")
                    
                except Exception as e:
                    print(f"Error testing {cat_var} vs {cont_var}: {str(e)}")
                    continue
        
        # Convert results to DataFrame
        if not results_list:
            print("Warning: No valid test results were generated.")
            return pd.DataFrame(columns=["Categorical", "Continuous", "Test", "Statistic", "p_value", 
                                         "Significant", "Groups", "DOF", "Effect_Size", "Effect_Type",
                                         "Power", "Normality", "Equal_Variance", 
                                         "Passed_Effect_Size_Filter", "Passed_Power_Filter", "Passed_Any_Filter"])
        
        results_df = pd.DataFrame(results_list)
        
        # Apply filtering criteria based on rules of thumb - with MORE LENIENT thresholds
        # Filter based on Power - REDUCED THRESHOLD
        for idx in results_df.index:
            if results_df.loc[idx, 'Power'] >= power_min:
                results_df.loc[idx, 'Passed_Power_Filter'] = True
        
        # Filter based on Effect Size (using SMALLER thresholds)
        for idx, row in results_df.iterrows():
            keep_row = True
            if row['Effect_Type'] == "Cohen's d":
                if abs(row['Effect_Size']) >= cohens_d_min:  # Small-to-medium effect for Cohen's d (reduced from 0.5)
                    results_df.loc[idx, 'Passed_Effect_Size_Filter'] = True
            elif row['Effect_Type'] == "Partial η²":
                if row['Effect_Size'] >= eta_squared_min:  # Small-to-medium effect for Partial η² (reduced from 0.06)
                    results_df.loc[idx, 'Passed_Effect_Size_Filter'] = True
            elif row['Effect_Type'] == "ε²":
                if row['Effect_Size'] >= epsilon_squared_min:  # Similar threshold as Partial η² (reduced from 0.06)
                    results_df.loc[idx, 'Passed_Effect_Size_Filter'] = True
            elif row['Effect_Type'] == "CLES":
                if abs(row['Effect_Size'] - 0.5) >= cles_diff_min:  # CLES should differ from 0.5 by at least 0.1
                    results_df.loc[idx, 'Passed_Effect_Size_Filter'] = True
        
        # Mark rows that pass any filter
        for idx in results_df.index:
            if results_df.loc[idx, 'Passed_Effect_Size_Filter'] or results_df.loc[idx, 'Passed_Power_Filter']:
                results_df.loc[idx, 'Passed_Any_Filter'] = True
            
            if results_df.loc[idx, 'Passed_Effect_Size_Filter'] and results_df.loc[idx, 'Passed_Power_Filter']:
                results_df.loc[idx, 'Passed_All_Filters'] = True
            
        
        # Print filtering results
        power_filtered = results_df[results_df['Passed_Power_Filter'] == True]
        effect_size_filtered = results_df[results_df['Passed_Effect_Size_Filter'] == True]
        any_filtered = results_df[results_df['Passed_Any_Filter'] == True]
        all_filtered = results_df[
            (results_df['Passed_Power_Filter'] == True) &
            (results_df['Passed_Effect_Size_Filter'] == True)

        ]
        
        print(f"\nFiltering Summary for ANOVA/t-tests:")
        print(f"Original results: {len(results_df)} tests")
        print(f"After Power filter (>= {power_min}): {len(power_filtered)} tests")
        print(f"After Effect Size filter (small-to-medium or larger): {len(effect_size_filtered)} tests")
        print(f"After any filter (passes at least one criterion): {len(any_filtered)} tests")
        print(f"After all filters (passes all criteria): {len(all_filtered)} tests")
        
        # IMPORTANT: Return ALL results with filter flags
        return results_df
    


    
    
    def correlation_test_wrapper(
            self, 
            data, 
            method    = 'pearson', 
            power_min = 0.6,
            corr_min  = 0.2,
            pairs     = None,
            directory = None,
        ):
        """
        Perform correlation tests for continuous-continuous relationships using pingouin.
        
        Args:
            data: DataFrame containing the variables
            pairs: List of tuples containing pairs of continuous variables
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            directory: Directory to save visualizations (if any)
        
        Returns:
            DataFrame containing test results
        """
        if pairs is None:
            all_pairs = self.get_pairs_of_variables(data)
            pairs = all_pairs["continuous_continuous"]
        
        results_list = []
        
        for var1, var2 in tqdm(pairs, desc=f"Running {method} correlation tests"):
            if var1 in data.columns and var2 in data.columns:
                try:
                    # Drop rows with NaN values
                    valid_data = data[[var1, var2]].dropna()
                    
                    if len(valid_data) < 3:
                        print(f"Skipping {var1} vs {var2}: Not enough data")
                        continue
                    
                    # Check normality if using Pearson correlation
                    normality_var1 = False
                    normality_var2 = False
                    
                    if method == 'pearson':
                        try:
                            # Use pingouin for normality test
                            norm_test_var1 = pg.normality(valid_data[var1])
                            norm_test_var2 = pg.normality(valid_data[var2])
                            
                            normality_var1 = norm_test_var1['normal'].values[0]
                            normality_var2 = norm_test_var2['normal'].values[0]
                            
                            if not (normality_var1 and normality_var2):
                                print(f"Warning: Non-normal data for {var1} or {var2}. Consider using Spearman correlation.")
                        except Exception as e:
                            print(f"Error in normality test for {var1} or {var2}: {str(e)}")
                    
                    # Use pingouin for correlation test
                    try:
                        corr_result = pg.corr(valid_data[var1], valid_data[var2], method=method)
                        
                        # Extract results
                        corr = corr_result['r'].values[0]
                        p_value = corr_result['p-val'].values[0]
                        n = corr_result['n'].values[0]
                        
                        # Calculate power
                        power = pg.power_corr(r=abs(corr), n=n, alpha=self.significance_threshold)
                        
                        # Check if result is significant
                        significant = p_value < self.significance_threshold
                        
                        # Add result to list
                        results_list.append({
                            "Variable1": var1,
                            "Variable2": var2,
                            "Method": method.capitalize(),
                            "Correlation": corr,
                            "p_value": p_value,
                            "Significant": significant,
                            "n": n,
                            "Power": power,
                            "Normality_Var1": normality_var1,
                            "Normality_Var2": normality_var2
                        })
                        
                        # Create visualization if directory provided
                        if directory and significant:
                            # Create scatter plot
                            plt.figure(figsize=(8, 6))
                            sns.scatterplot(x=var1, y=var2, data=valid_data)
                            plt.title(f'Relationship between {var1} and {var2}\n'
                                     f'({method.capitalize()} r={corr:.2f}, p={p_value:.4f}, n={n})')
                            
                            # Add regression line
                            sns.regplot(x=var1, y=var2, data=valid_data, scatter=False, line_kws={"color": "red"})
                            
                            plt.tight_layout()
                            plt.savefig(f"{directory}/{var1}_{var2}_scatter.png")
                            plt.close()
                            
                            # Create joint plot (includes distributions) with fixed title
                            try:
                                # Create the joint plot
                                g = sns.jointplot(x=var1, y=var2, data=valid_data, kind="reg")
                                
                                # Create a figure-level title that's visible (higher y position)
                                plt.subplots_adjust(top=0.9)  # Make room for the title
                                plt.suptitle(f'Relationship between {var1} and {var2}', fontsize=12, y=0.98)
                                
                                # Add correlation details as text inside the plot instead of in the title
                                # Get the joint axes (main scatter plot)
                                ax = g.ax_joint
                                ax.text(0.05, 0.95, f'{method.capitalize()} r={corr:.2f}\np={p_value:.4f}, n={n}',
                                      transform=ax.transAxes, fontsize=10, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                                
                                plt.tight_layout()
                                plt.savefig(f"{directory}/{var1}_{var2}_jointplot.png", bbox_inches='tight', dpi=300)
                                plt.close()
                            except Exception as viz_err:
                                print(f"Error creating joint plot for {var1} vs {var2}: {str(viz_err)}")
                        
                    except Exception as corr_err:
                        print(f"Error in correlation test for {var1} vs {var2}: {str(corr_err)}")
                        continue
                    
                except Exception as e:
                    print(f"Error testing {var1} vs {var2}: {str(e)}")
                    continue
        
        # Convert results to DataFrame
        if not results_list:
            print("Warning: No valid test results were generated.")
            return pd.DataFrame(columns=["Variable1", "Variable2", "Method", "Correlation", "p_value",
                                        "Significant", "n", "Power", "Normality_Var1", "Normality_Var2"])
        
        results_df = pd.DataFrame(results_list)
        
        # Apply filtering criteria based on rules of thumb - but with more lenient thresholds
        # Store the original results for reference
        original_results = results_df.copy()
        
        # Filter based on correlation strength - REDUCED THRESHOLD
        corr_filtered = results_df[abs(results_df['Correlation']) >= corr_min]
        
        # Filter based on Power - REDUCED THRESHOLD
        power_filtered = results_df[results_df['Power'] >= power_min]
        
        # Combine filters with OR instead of AND to be more inclusive
        combined_filtered = pd.concat([corr_filtered, power_filtered]).drop_duplicates()

        # Combine filters AND to be more inclusive
        all_filtered = results_df[
            (results_df['Power'] >= power_min) & 
            (abs(results_df['Correlation']) >= corr_min)
        ]
        
        # Print filtering results
        print(f"\nFiltering Summary for {method.capitalize()} Correlation Tests:")
        print(f"Original results: {len(results_df)} tests")
        print(f"After Correlation Strength filter (|r| >= {corr_min}): {len(corr_filtered)} tests")
        print(f"After Power filter (>= {power_min}): {len(power_filtered)} tests")
        print(f"After both filters : {len(all_filtered)} tests")
        print(f"After combined filters (less strict): {len(combined_filtered)} tests")
        
        # Include ALL results in the output DataFrame
        results_df['Passed_Correlation_Filter'] = False
        results_df['Passed_Power_Filter'] = False
        results_df['Passed_Any_Filter'] = False
        results_df['Passed_All_Filters'] = False
        
        for idx in results_df.index:
            # Mark which filters each result passes
            if abs(results_df.loc[idx, 'Correlation']) >= corr_min:
                results_df.loc[idx, 'Passed_Correlation_Filter'] = True
            
            if results_df.loc[idx, 'Power'] >= power_min:
                results_df.loc[idx, 'Passed_Power_Filter'] = True
            
            # Mark if it passed any filter
            if idx in combined_filtered.index:
                results_df.loc[idx, 'Passed_Any_Filter'] = True
            
            # Mark if it passed all filters
            if idx in all_filtered.index:
                results_df.loc[idx, 'Passed_All_Filters'] = True
        
        # IMPORTANT: Return ALL results, not just filtered ones
        return results_df
    
    # def nonparametric_group_test_wrapper(self, data, pairs=None, directory=None):
    #     """
    #     Perform nonparametric tests for categorical-continuous relationships using pingouin.
        
    #     Args:
    #         data: DataFrame containing the variables
    #         pairs: List of tuples containing pairs of (categorical, continuous) variables
    #         directory: Directory to save visualizations (if any)
        
    #     Returns:
    #         DataFrame containing test results
    #     """
    #     if pairs is None:
    #         all_pairs = self.get_pairs_of_variables(data)
    #         pairs = all_pairs["categorical_continuous"]
        
    #     results_list = []
        
    #     for cat_var, cont_var in tqdm(pairs, desc="Running nonparametric group tests"):
    #         if cat_var in data.columns and cont_var in data.columns:
    #             try:
    #                 # Create a filtered dataframe with no NaN values
    #                 filtered_data = data[[cat_var, cont_var]].dropna()
                    
    #                 # Skip if not enough data
    #                 if len(filtered_data) < 3:
    #                     print(f"Skipping {cat_var} vs {cont_var}: Not enough data")
    #                     continue
                    
    #                 # Get unique categories
    #                 categories = filtered_data[cat_var].unique()
                    
    #                 if len(categories) < 2:
    #                     print(f"Skipping {cat_var} vs {cont_var}: Need at least 2 categories")
    #                     continue
                    
    #                 # Prepare groups for analysis
    #                 groups = []
    #                 group_sizes = []
    #                 group_data_by_category = {}
                    
    #                 for category in categories:
    #                     group_data = filtered_data[filtered_data[cat_var] == category][cont_var]
    #                     if len(group_data) >= 2:  # Need at least 2 samples per group
    #                         groups.append(group_data.values)
    #                         group_sizes.append(len(group_data))
    #                         group_data_by_category[category] = group_data.values
                    
    #                 if len(groups) < 2:
    #                     print(f"Skipping {cat_var} vs {cont_var}: Not enough valid groups")
    #                     continue
                    
    #                 # Choose appropriate test based on number of categories
    #                 if len(groups) == 2:
    #                     # Two groups: Mann-Whitney U test
    #                     try:
    #                         group_values = list(group_data_by_category.values())
    #                         mw_result = pg.mwu(group_values[0], group_values[1])
    #                         test_used = "Mann-Whitney U"
    #                         statistic = mw_result['U-val'].values[0]
    #                         p_value = mw_result['p-val'].values[0]
                            
    #                         # Calculate effect size (CLES - Common Language Effect Size)
    #                         effect_size = mw_result['CLES'].values[0]
    #                         effect_type = "CLES"
                            
    #                         # Calculate rank-biserial correlation (another effect size)
    #                         n1, n2 = len(group_values[0]), len(group_values[1])
    #                         n_total = n1 + n2
    #                         rank_biserial = 1 - (2 * statistic) / (n1 * n2)
                            
    #                         # Approximate power calculation for Mann-Whitney U test
    #                         # Convert to equivalent Cohen's d for estimation
    #                         means = [np.mean(g) for g in group_values]
    #                         pooled_std = np.sqrt(np.sum([(len(g) - 1) * np.var(g) for g in group_values]) / (n_total - 2))
    #                         equiv_d = abs(means[0] - means[1]) / pooled_std if pooled_std > 0 else 0
                            
    #                         # Adjust for non-parametric efficiency (~0.95 for large samples)
    #                         equiv_d_adjusted = equiv_d * 0.95  
    #                         power = pg.power_ttest2n(d=equiv_d_adjusted, nx=n1, ny=n2, alpha=self.significance_threshold)
                            
    #                     except Exception as e:
    #                         print(f"Error in Mann-Whitney U test for {cat_var} vs {cont_var}: {str(e)}")
    #                         continue
                            
    #                 else:
    #                     # More than two groups: Kruskal-Wallis test
    #                     try:
    #                         kw_result = pg.kruskal(dv=cont_var, between=cat_var, data=filtered_data)
    #                         test_used = "Kruskal-Wallis"
    #                         statistic = kw_result['H'].values[0]
    #                         p_value = kw_result['p-unc'].values[0]
                            
    #                         # Calculate effect size (epsilon-squared)
    #                         n = filtered_data.shape[0]
    #                         k = len(groups)
    #                         dof = k - 1
    #                         epsilon_squared = (statistic - dof) / (n - k)
    #                         effect_size = max(0, epsilon_squared)  # Ensure non-negative
    #                         effect_type = "ε²"
    #                         rank_biserial = np.nan  # Not applicable for >2 groups
                            
    #                         # Approximate power calculation for Kruskal-Wallis
    #                         # Convert epsilon-squared to Cohen's f for power approximation
    #                         equiv_f = np.sqrt(effect_size / (1 - effect_size)) if effect_size < 1 else 1.0
    #                         power = pg.power_anova(eta_squared=effect_size, k=k, n=n, alpha=self.significance_threshold)
                            
    #                     except Exception as e:
    #                         print(f"Error in Kruskal-Wallis test for {cat_var} vs {cont_var}: {str(e)}")
    #                         continue
                    
    #                 # Check if result is significant
    #                 significant = p_value < self.significance_threshold
                    
    #                 # Add result to list
    #                 results_list.append({
    #                     "Categorical": cat_var,
    #                     "Continuous": cont_var,
    #                     "Test": test_used,
    #                     "Statistic": statistic,
    #                     "p_value": p_value,
    #                     "Significant": significant,  # Explicitly include the Significant column
    #                     "Groups": len(groups),
    #                     "Total_n": sum(group_sizes),
    #                     "Effect_Size": effect_size,
    #                     "Effect_Type": effect_type,
    #                     "Rank_Biserial": rank_biserial if 'rank_biserial' in locals() else np.nan,
    #                     "Power": power
    #                 })
                    
    #                 # Perform post-hoc tests if significant and more than 2 groups
    #                 if significant and len(groups) > 2:
    #                     try:
    #                         # Non-parametric post-hoc: Dunn's test
    #                         posthoc = pg.pairwise_tests(dv=cont_var, between=cat_var, data=filtered_data, 
    #                                                     parametric=False, padjust='bonf')
                            
    #                         # Save post-hoc results
    #                         if directory:
    #                             posthoc.to_csv(f"{directory}/{cat_var}_{cont_var}_posthoc_dunn.csv", index=False)
                                
    #                     except Exception as posthoc_err:
    #                         print(f"Error in post-hoc test for {cat_var} vs {cont_var}: {str(posthoc_err)}")
                    
    #                 # Create visualization if directory provided
    #                 if directory and significant:
    #                     try:
    #                         # Create boxplot visualization
    #                         plt.figure(figsize=(12, 6))
    #                         sns.boxplot(x=cat_var, y=cont_var, data=filtered_data)
    #                         plt.title(f'Distribution of {cont_var} by {cat_var}\n({test_used}, p={p_value:.4f})')
    #                         plt.xticks(rotation=45)
    #                         plt.tight_layout()
    #                         plt.savefig(f"{directory}/{cat_var}_{cont_var}_nonparametric_boxplot.png")
    #                         plt.close()
                            
    #                         # Create violin plot for better visualization of distributions
    #                         plt.figure(figsize=(12, 6))
    #                         sns.violinplot(x=cat_var, y=cont_var, data=filtered_data, inner="points")
    #                         plt.title(f'Distribution of {cont_var} by {cat_var}\n({test_used}, p={p_value:.4f})')
    #                         plt.xticks(rotation=45)
    #                         plt.tight_layout()
    #                         plt.savefig(f"{directory}/{cat_var}_{cont_var}_nonparametric_violin.png")
    #                         plt.close()
                            
    #                     except Exception as viz_err:
    #                         print(f"Error creating visualization for {cat_var} vs {cont_var}: {str(viz_err)}")
                    
    #             except Exception as e:
    #                 print(f"Error testing {cat_var} vs {cont_var}: {str(e)}")
    #                 continue
        
    #     # Convert results to DataFrame
    #     if not results_list:
    #         print("Warning: No valid test results were generated.")
    #         return pd.DataFrame(columns=["Categorical", "Continuous", "Test", "Statistic", "p_value", 
    #                                      "Significant", "Groups", "Total_n", "Effect_Size", "Effect_Type", 
    #                                      "Rank_Biserial", "Power"])
        
    #     results_df = pd.DataFrame(results_list)
        
    #     # Apply filtering criteria based on rules of thumb
    #     # Store the original results for reference
    #     original_results = results_df.copy()
        
    #     # Filter based on Effect Size
    #     # Different effect size measures require different thresholds
    #     effect_size_filtered = results_df.copy()
    #     for idx, row in results_df.iterrows():
    #         keep_row = True
    #         if row['Effect_Type'] == "CLES":
    #             if abs(row['Effect_Size'] - 0.5) < 0.1:  # CLES should differ from 0.5 by at least 0.1
    #                 keep_row = False
    #         elif row['Effect_Type'] == "ε²":
    #             if row['Effect_Size'] < 0.06:  # Similar threshold as for ANOVA
    #                 keep_row = False
            
    #         if not keep_row:
    #             effect_size_filtered = effect_size_filtered.drop(idx)
        
    #     # Filter based on Power
    #     min_power = 0.8  # Conventional threshold for adequate power
    #     power_filtered = results_df[results_df['Power'] >= min_power]
        
    #     # Combine filters
    #     combined_filtered = pd.merge(effect_size_filtered, power_filtered, how='inner')
        
    #     # Print filtering results
    #     print(f"\nFiltering Summary for Nonparametric Group Tests:")
    #     print(f"Original results: {len(results_df)} tests")
    #     print(f"After Effect Size filter (medium or larger): {len(effect_size_filtered)} tests")
    #     print(f"After Power filter (>= {min_power}): {len(power_filtered)} tests")
    #     print(f"After combined filters: {len(combined_filtered)} tests")
        
    #     # If all filtering leaves no results, return the original with a warning
    #     if len(combined_filtered) == 0:
    #         print("Warning: Filtering criteria removed all results. Returning original results.")
    #         # Add a column indicating which results passed all filters
    #         original_results['Passed_All_Filters'] = False
    #         return original_results
        
    #     # Add a column indicating which results passed all filters
    #     results_df['Passed_All_Filters'] = False
    #     for idx in combined_filtered.index:
    #         if idx in results_df.index:  # Make sure the index exists
    #             results_df.loc[idx, 'Passed_All_Filters'] = True
        
    #     return results_df


class DescriptiveStatisticsAnalyzer:
    """Performs descriptive statistics and visualizations on survey data."""
    
    def __init__(self, data):
        self.data = data
        self.prepare_data()

        
    def prepare_data(self):
        """Prepare data for analysis by converting types and creating necessary columns."""
        # Convert time columns to datetime
        if 'Start Time' in self.data.columns and 'End Time' in self.data.columns:
            self.data['Start Time'] = pd.to_datetime(self.data['Start Time'])
            self.data['End Time'] = pd.to_datetime(self.data['End Time'])
            self.data['Duration'] = (self.data['End Time'] - self.data['Start Time']).dt.total_seconds() / 60  # in minutes
        
        # Create Has_Email column
        self.data['Has_Email'] = self.data['Email'].notna()
        
        # Reshape the data for question analysis
        self.question_data = self.reshape_for_questions()
    
    def reshape_for_questions(self):
        """Reshape data to have questions as separate rows for easier analysis."""
        columns_to_keep = ['Age Group', 'Gender', 'Educational Level', 'Employment Status', 
                         'Living Situation', 'Physical Activity Level', 'City', 'Has_Email']
        
        # Get all question columns (Q1-Q10)
        question_cols = [col for col in self.data.columns if col.startswith('Q')]
        
        # Create a new dataframe with reshaped data
        reshaped_data = []
        
        for _, row in self.data.iterrows():
            for q_col in question_cols:
                new_row = {col: row[col] for col in columns_to_keep}
                new_row['Question'] = q_col
                new_row['Answer'] = row[q_col]
                reshaped_data.append(new_row)
                
        return pd.DataFrame(reshaped_data)
    
    def analyze(self):
        """Return basic descriptive statistics for numerical columns."""
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        return self.data[numeric_cols].describe().T
    
    def unique_values(self):
        """Count unique values in each column."""
        return pd.DataFrame({
            'Column': self.data.columns,
            'Unique Values': [self.data[col].nunique() for col in self.data.columns]
        }).set_index('Column')
    
    def total_duration(self):
        """Calculate survey completion time statistics."""
        duration_stats = {
            'Average Duration (minutes)': self.data['Duration'].mean(),
            'Median Duration (minutes)': self.data['Duration'].median(),
            'Min Duration (minutes)': self.data['Duration'].min(),
            'Max Duration (minutes)': self.data['Duration'].max()
        }
        return pd.Series(duration_stats)
    
    def anonymity_count(self):
        """Count anonymous vs non-anonymous responses."""
        anonymous = self.data['Email'].isna().sum()
        non_anonymous = self.data['Email'].notna().sum()
        return pd.Series({
            'Anonymous': anonymous,
            'Non-Anonymous': non_anonymous,
            'Percentage Anonymous': anonymous / (anonymous + non_anonymous) * 100
        })
    
    def descriptive_statistics_by_question(self):
        """Calculate descriptive statistics for each question."""
        return self.question_data.groupby('Question')['Answer'].describe()
    
    def analyze_by_group(self, group_column):
        """Analyze responses by a specific grouping variable."""
        if group_column not in self.question_data.columns:
            return f"Column {group_column} not found in data"
        
        return self.question_data.groupby([group_column, 'Question'])['Answer'].describe()
    
    def plot_histograms(self):
        """Plot histograms for all questions."""
        questions = sorted(self.question_data['Question'].unique())
        num_questions = len(questions)
        
        # Calculate rows and columns for subplots
        cols = min(3, num_questions)
        rows = (num_questions + cols - 1) // cols
        
        plt.figure(figsize=(15, rows * 4))
        
        for i, question in enumerate(questions, 1):
            plt.subplot(rows, cols, i)
            
            # Get question data and plot histogram
            q_data = self.question_data[self.question_data['Question'] == question]['Answer']
            sns.histplot(q_data, bins=6, kde=True)
            
            plt.title(f'{question} Distribution')
            plt.xlabel('Response (0-5 scale)')
            plt.ylabel('Count')
            plt.xlim(-0.5, 5.5)
            
        plt.tight_layout()
        return plt.gcf()
    
    def plot_boxplots(self):
        """Plot boxplots comparing all questions."""
        plt.figure(figsize=(12, 6))
        
        # Create boxplot
        sns.boxplot(x='Question', y='Answer', data=self.question_data, palette='viridis')
        
        plt.title('Response Distribution by Question')
        plt.xlabel('Question')
        plt.ylabel('Response (0-5 scale)')
        plt.xticks(rotation=45)
        plt.ylim(-0.5, 5.5)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap between all questions."""
        # Get question columns
        question_cols = [col for col in self.data.columns if col.startswith('Q')]
        
        # Calculate correlation matrix
        corr_matrix = self.data[question_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                   vmin=-1, vmax=1, square=True)
        
        plt.title('Correlation Between Questions')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_response_by_demographic(self, demographic_col):
        """Plot average responses by demographic variable."""
        if demographic_col not in self.question_data.columns:
            return f"Column {demographic_col} not found in data"
        
        # Calculate mean response by demographic and question
        grouped_data = self.question_data.groupby([demographic_col, 'Question'])['Answer'].mean().reset_index()
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Create pivot table for easier plotting
        pivot_data = grouped_data.pivot(index=demographic_col, columns='Question', values='Answer')
        
        # Plot heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlGnBu')
        
        plt.title(f'Average Responses by {demographic_col}')
        plt.ylabel(demographic_col)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_duration_histogram(self):
        """Plot histogram of survey completion duration."""
        plt.figure(figsize=(10, 6))
        
        sns.histplot(self.data['Duration'], bins=15, kde=True)
        
        plt.title('Survey Completion Time Distribution')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Count')
        
        # Add vertical line for mean and median
        plt.axvline(self.data['Duration'].mean(), color='red', linestyle='--', label=f'Mean: {self.data["Duration"].mean():.2f} min')
        plt.axvline(self.data['Duration'].median(), color='green', linestyle='--', label=f'Median: {self.data["Duration"].median():.2f} min')
        
        plt.legend()
        plt.tight_layout()
        return plt.gcf()
    
    def plot_response_patterns(self):
        """Create a heatmap showing response patterns across all participants."""
        # Get question columns
        question_cols = [col for col in self.data.columns if col.startswith('Q')]
        
        # Create heatmap of responses
        plt.figure(figsize=(12, 10))
        
        # Use a subset if there are too many respondents
        max_display = 30
        data_subset = self.data[question_cols].iloc[:max_display] if len(self.data) > max_display else self.data[question_cols]
        
        sns.heatmap(data_subset, cmap='YlGnBu', annot=True, fmt='.0f', 
                   xticklabels=question_cols, 
                   yticklabels=[f"Resp {i+1}" for i in range(len(data_subset))])
        
        plt.title('Response Patterns by Participant')
        plt.xlabel('Question')
        plt.ylabel('Participant')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_city_distribution(self):
        """Plot distribution of responses by city."""
        plt.figure(figsize=(12, 6))
        
        # Count responses by city
        city_counts = self.data['City'].value_counts()
        
        # Create bar plot
        sns.barplot(x=city_counts.index, y=city_counts.values, palette='viridis')
        
        plt.title('Number of Responses by City')
        plt.xlabel('City')
        plt.ylabel('Number of Responses')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_age_gender_distribution(self):
        """Plot age distribution by gender."""
        plt.figure(figsize=(12, 6))
        
        # Create crosstab
        age_gender = pd.crosstab(self.data['Age Group'], self.data['Gender'])
        
        # Plot stacked bar chart
        age_gender.plot(kind='bar', stacked=True, colormap='viridis')
        
        plt.title('Age Distribution by Gender')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.legend(title='Gender')
        plt.tight_layout()
        return plt.gcf()
    
    def run_all(self):
        """Run all analyses and return results in a dictionary."""
        results = {
            "Basic Statistics": self.analyze(),
            "Unique Values": self.unique_values(),
            "Survey Duration": self.total_duration(),
            "Anonymity Count": self.anonymity_count(),
            "Question Statistics": self.descriptive_statistics_by_question(),
            "Gender Analysis": self.analyze_by_group('Gender'),
            "Education Analysis": self.analyze_by_group('Educational Level'),
            "Histograms": self.plot_histograms(),
            "Boxplots": self.plot_boxplots(),
            "Correlation Heatmap": self.plot_correlation_heatmap(),
            "Gender Responses": self.plot_response_by_demographic('Gender'),
            "Education Responses": self.plot_response_by_demographic('Educational Level'),
            "Duration Histogram": self.plot_duration_histogram(),
            "Response Patterns": self.plot_response_patterns(),
            "City Distribution": self.plot_city_distribution(),
            "Age Gender Distribution": self.plot_age_gender_distribution()
        }
        return results

class DescriptivePostStatisticsAnalyzer:
    """Performs descriptive statistics and visualizations on survey data."""
    
    def __init__(self):
        pass

    def create_summary_visualizations(
            self, 
            data_df   : pd.DataFrame,
            directory : str
        ):
        # Make a copy to avoid modifying the original
        df = data_df.copy()
        
        # Clean up any duplicate entries (we see some in the data)
        df = df.drop_duplicates()
        
        # Make sure question columns are numeric
        question_cols = [f'Q{i}' for i in range(1, 11)]
        for col in question_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Analysis
        analyzer = DescriptiveStatisticsAnalyzer(df)
        
        # Create additional custom visualizations
        
        # 1. Response patterns visualization
        plt.figure(figsize=(14, 10))
        question_cols = [f'Q{i}' for i in range(1, 11)]
        response_data = df[question_cols].copy()
        
        # Create a heatmap with annotations
        sns.heatmap(response_data.head(20), cmap='YlGnBu', annot=True, fmt='.0f', 
                xticklabels=question_cols, 
                yticklabels=[f"Resp {i+1}" for i in range(20)])
        
        plt.title('Individual Response Patterns (First 20 Respondents)')
        plt.xlabel('Question')
        plt.ylabel('Respondent')
        plt.tight_layout()
        plt.savefig(f'{directory}/individual_responses.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Question mean responses
        plt.figure(figsize=(10, 6))
        means = [df[col].mean() for col in question_cols]
        std_errs = [df[col].std() / np.sqrt(len(df)) for col in question_cols]
        
        plt.bar(question_cols, means, yerr=std_errs, capsize=5, color='skyblue')
        plt.axhline(y=2.5, color='r', linestyle='--', alpha=0.7, label='Midpoint (2.5)')
        
        plt.title('Average Response by Question')
        plt.xlabel('Question')
        plt.ylabel('Average Score (0-5 scale)')
        plt.ylim(0, 5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{directory}/question_means.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Physical activity vs response
        plt.figure(figsize=(12, 8))
        activity_levels = df['Physical Activity Level'].unique()
        
        activity_data = []
        for q in question_cols:
            for activity in activity_levels:
                avg_response = df[df['Physical Activity Level'] == activity][q].mean()
                activity_data.append({
                    'Question': q,
                    'Activity Level': activity,
                    'Average Response': avg_response
                })
        
        activity_df = pd.DataFrame(activity_data)
        
        # Create pivot table for the heatmap
        pivot_activity = activity_df.pivot(index='Activity Level', columns='Question', values='Average Response')
        
        # Plot heatmap
        sns.heatmap(pivot_activity, annot=True, fmt='.1f', cmap='viridis')
        plt.title('Average Responses by Physical Activity Level')
        plt.tight_layout()
        plt.savefig(f'{directory}/activity_responses.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print overall summary
        print("\nSurvey Analysis Complete!")
        print("Generated visualizations saved as PNG files.")
        print("Key insights:")
        
        # Calculate high/low questions
        question_means = {q: df[q].mean() for q in question_cols}
        highest_q = max(question_means, key=question_means.get)
        lowest_q = min(question_means, key=question_means.get)
        
        print(f"- Highest scoring question: {highest_q} (avg: {question_means[highest_q]:.2f})")
        print(f"- Lowest scoring question: {lowest_q} (avg: {question_means[lowest_q]:.2f})")
        
        # Gender differences
        male_responses = df[df['Gender'] == 'Male'][question_cols].mean()
        female_responses = df[df['Gender'] == 'Female'][question_cols].mean()
        
        biggest_diff_q = max(question_cols, key=lambda q: abs(male_responses[q] - female_responses[q]))
        diff_value = abs(male_responses[biggest_diff_q] - female_responses[biggest_diff_q])
        
        print(f"- Largest gender difference: {biggest_diff_q} (diff: {diff_value:.2f})")
        
        return analyzer

    def create_distribution_visualizations(
            self, 
            data_df   : pd.DataFrame,
            directory : str
        ):
        """
        Process the survey data with questions in columns Q1-Q10 format.
        
        Args:
            data_df: DataFrame with the survey data
        
        Returns:
            analyzer: The DescriptiveStatisticsAnalyzer object with processed data
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Make a copy to avoid modifying the original
        df = data_df.copy()
        
        # Clean up any duplicate entries (we see some in the data)
        df = df.drop_duplicates()
        
        # Make sure question columns are numeric
        question_cols = [f'Q{i}' for i in range(1, 11)]
        for col in question_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Analysis
        analyzer = DescriptiveStatisticsAnalyzer(df)
        
        # Generate and display key visualizations
        plt.figure(figsize=(12, 8))
        
        # 1. Question Response Distributions
        fig1 = analyzer.plot_histograms()
        plt.savefig(f'{directory}/question_distributions.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. Boxplot Comparison
        fig2 = analyzer.plot_boxplots()
        plt.savefig(f'{directory}/question_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. Correlation Heatmap
        fig3 = analyzer.plot_correlation_heatmap()
        plt.savefig(f'{directory}/question_correlations.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # 4. Demographic Analysis
        fig4 = analyzer.plot_response_by_demographic('Gender')
        plt.savefig(f'{directory}/gender_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        fig5 = analyzer.plot_response_by_demographic('Educational Level')
        plt.savefig(f'{directory}/education_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        fig6 = analyzer.plot_age_gender_distribution()
        plt.savefig(f'{directory}/age_gender_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig6)
        
        # 5. City Distribution
        fig7 = analyzer.plot_city_distribution()
        plt.savefig(f'{directory}/city_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig7)
        
        # Print summary statistics
        print("\nSurvey Summary Statistics:")
        print(f"Total Responses: {len(df)}")
        print(f"Anonymous Rate: {analyzer.anonymity_count()['Percentage Anonymous']:.1f}%")
        print(f"Average Completion Time: {analyzer.total_duration()['Average Duration (minutes)']:.2f} minutes")
        
        print("\nQuestion Averages:")
        q_means = {q: df[q].mean() for q in question_cols if q in df.columns}
        for q, mean in q_means.items():
            print(f"{q}: {mean:.2f}")
        
        return analyzer

    def create_demographic_visualizations(
            self, 
            data_df   : pd.DataFrame,
            directory : str
        ):
        """Create visualizations focused on demographic information."""

        # Make a copy to avoid modifying the original
        df = data_df.copy()
        
        # 1. Age Group Distribution
        plt.figure(figsize=(10, 6))
        age_counts = df['Age Group'].value_counts().sort_index()
        
        sns.barplot(x=age_counts.index, y=age_counts.values, palette='viridis')
        
        for i, count in enumerate(age_counts.values):
            plt.text(i, count + 0.5, str(count), ha='center')
        
        plt.title('Age Distribution of Survey Respondents')
        plt.xlabel('Age Group')
        plt.ylabel('Number of Respondents')
        plt.tight_layout()
        plt.savefig(f'{directory}/age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Education Level Distribution
        plt.figure(figsize=(12, 6))
        edu_counts = df['Educational Level'].value_counts().sort_values()
        
        sns.barplot(x=edu_counts.values, y=edu_counts.index, palette='Blues_r')
        
        for i, count in enumerate(edu_counts.values):
            plt.text(count + 0.5, i, str(count), va='center')
        
        plt.title('Education Level of Survey Respondents')
        plt.xlabel('Number of Respondents')
        plt.ylabel('Education Level')
        plt.tight_layout()
        plt.savefig(f'{directory}/education_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Employment Status
        plt.figure(figsize=(12, 6))
        employment_counts = df['Employment Status'].value_counts()
        
        # Create a colormap
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(employment_counts)))
        
        # Create pie chart
        plt.pie(employment_counts, labels=employment_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90, shadow=False, wedgeprops={'edgecolor': 'white'})
        
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Employment Status of Survey Respondents')
        plt.tight_layout()
        plt.savefig(f'{directory}/employment_status.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Living Situation vs Physical Activity Level
        plt.figure(figsize=(14, 8))
        
        # Create crosstab
        living_activity = pd.crosstab(
            df['Living Situation'], 
            df['Physical Activity Level']
        )
        
        # Create a heatmap
        sns.heatmap(living_activity, annot=True, fmt='d', cmap='YlGnBu')
        
        plt.title('Living Situation vs Physical Activity Level')
        plt.tight_layout()
        plt.savefig(f'{directory}/living_activity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. City Distribution (Top 5)
        plt.figure(figsize=(12, 6))
        city_counts = df['City'].value_counts().head(5)
        
        sns.barplot(x=city_counts.index, y=city_counts.values, palette='viridis')
        
        for i, count in enumerate(city_counts.values):
            plt.text(i, count + 0.3, str(count), ha='center')
        
        plt.title('Top 5 Cities of Survey Respondents')
        plt.xlabel('City')
        plt.ylabel('Number of Respondents')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{directory}/top_cities.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Question response patterns by demographic factors
        question_cols = [f'Q{i}' for i in range(1, 11)]
        
        # Create a function to generate demographic comparison charts
        def plot_question_by_demographic(demographic_col, title_prefix, filename_prefix):
            plt.figure(figsize=(14, 8))
            
            # Get unique values in the demographic column
            unique_values = df[demographic_col].unique()
            
            # Set up colors
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
            
            # Create grouped bar chart
            bar_width = 0.8 / len(unique_values)
            
            for i, value in enumerate(unique_values):
                # Get means for this demographic group
                group_data = df[df[demographic_col] == value]
                means = [group_data[q].mean() for q in question_cols]
                
                # Calculate positions for bars
                positions = np.arange(len(question_cols)) + (i - len(unique_values)/2 + 0.5) * bar_width
                
                plt.bar(positions, means, width=bar_width, color=colors[i], 
                    label=f'{value}')
            
            # Add details
            plt.axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='Midpoint')
            plt.xlabel('Question')
            plt.ylabel('Average Score (0-5)')
            plt.title(f'{title_prefix} Comparison by Question')
            plt.xticks(np.arange(len(question_cols)), question_cols)
            plt.ylim(0, 5)
            plt.legend(title=demographic_col)
            plt.tight_layout()
            
            plt.savefig(f'{directory}/{filename_prefix}_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create demographic comparisons
        plot_question_by_demographic('Gender', 'Gender', 'gender')
        plot_question_by_demographic('Physical Activity Level', 'Activity Level', 'activity')
        
        print("Demographic visualizations created successfully.")


    def create_advanced_visualizations(
            self, 
            data_df   : pd.DataFrame,
            directory : str
        ):
        """
        Create 10 advanced visualizations for deeper analysis of the survey data.
        
        Args:
            data_df: DataFrame with the survey data
            directory: Directory to save the visualizations
        """

        # Make a copy to avoid modifying the original
        df = data_df.copy()
        
        # Clean up any duplicate entries
        df = df.drop_duplicates()
        
        # Make sure question columns are numeric
        question_cols = [f'Q{i}' for i in range(1, 11)]
        for col in question_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. Response Distribution by Question (Violin Plot)
        plt.figure(figsize=(14, 8))
        
        # Melt the data for seaborn
        melted_data = pd.melt(df, id_vars=['Gender'], value_vars=question_cols, 
                              var_name='Question', value_name='Response')
        
        # Create violin plot
        sns.violinplot(x='Question', y='Response', data=melted_data, palette='viridis', inner='quartile')
        plt.title('Response Distribution by Question (Violin Plot)')
        plt.xlabel('Question')
        plt.ylabel('Response Value (0-5)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'{directory}/response_violins.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Question Response Correlation Network
        plt.figure(figsize=(10, 10))
        
        # Calculate correlation matrix
        corr_matrix = df[question_cols].corr()
        
        # Filter only stronger correlations
        threshold = 0.2
        
        # Create positions for nodes in a circle
        num_nodes = len(question_cols)
        angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
        
        # Plot nodes
        node_x = np.cos(angles)
        node_y = np.sin(angles)
        plt.scatter(node_x, node_y, s=500, color='skyblue', zorder=3)
        
        # Plot node labels
        for i, question in enumerate(question_cols):
            plt.text(node_x[i]*1.1, node_y[i]*1.1, question, ha='center', va='center', 
                     fontsize=12, fontweight='bold')
        
        # Plot edges
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) >= threshold:
                    # Line width proportional to correlation strength
                    line_width = abs(correlation) * 5
                    # Line color based on positive/negative correlation
                    line_color = 'green' if correlation > 0 else 'red'
                    alpha = min(1, abs(correlation) + 0.2)
                    
                    plt.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]], 
                             color=line_color, linewidth=line_width, alpha=alpha, zorder=1)
        
        # Create a legend
        legend_elements = [
            Patch(facecolor='green', edgecolor='green', label='Positive Correlation'),
            Patch(facecolor='red', edgecolor='red', label='Negative Correlation')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.title('Question Correlation Network\n(Thicker lines = stronger correlations)', fontsize=14)
        plt.axis('equal')
        plt.axis('off')
        plt.savefig(f'{directory}/correlation_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Response Patterns by Age Group
        plt.figure(figsize=(14, 8))
        
        # Aggregate data by age group
        age_response = df.groupby('Age Group')[question_cols].mean().reset_index()
        
        # Create heatmap
        pivoted = age_response.set_index('Age Group')
        sns.heatmap(pivoted, annot=True, fmt='.1f', cmap='YlGnBu')
        
        plt.title('Average Response by Age Group')
        plt.ylabel('Age Group')
        plt.xlabel('Question')
        plt.savefig(f'{directory}/age_response_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Survey Completion Time Analysis
        plt.figure(figsize=(12, 6))
        
        # Convert timestamp columns to datetime
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])
        
        # Calculate completion time in minutes
        df['Completion Time'] = (df['End Time'] - df['Start Time']).dt.total_seconds() / 60
        
        # Plot histogram of completion times
        sns.histplot(df['Completion Time'], bins=15, kde=True, color='purple')
        plt.axvline(df['Completion Time'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["Completion Time"].mean():.2f} minutes')
        plt.axvline(df['Completion Time'].median(), color='green', linestyle='-', 
                    label=f'Median: {df["Completion Time"].median():.2f} minutes')
        
        plt.title('Survey Completion Time Distribution')
        plt.xlabel('Completion Time (minutes)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(f'{directory}/completion_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Radar Chart of Questions by Gender
        plt.figure(figsize=(10, 10))
        
        # Calculate mean responses by gender
        gender_means = df.groupby('Gender')[question_cols].mean()
        
        # Set up radar chart
        categories = question_cols
        N = len(categories)
        
        # Create the angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Initialize the subplot
        ax = plt.subplot(111, polar=True)
        
        # Draw one line per gender
        for gender, color in zip(['Male', 'Female'], ['blue', 'red']):
            if gender in gender_means.index:
                values = gender_means.loc[gender].values.tolist()
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, 'o-', linewidth=2, label=gender, color=color)
                ax.fill(angles, values, alpha=0.1, color=color)
        
        # Set labels and clean up
        plt.xticks(angles[:-1], categories)
        plt.yticks([1, 2, 3, 4, 5], color='gray')
        plt.ylim(0, 5)
        
        plt.title('Question Responses by Gender (Radar Chart)')
        plt.legend(loc='upper right')
        plt.savefig(f'{directory}/gender_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Response Patterns by City (Top 5 Cities)
        plt.figure(figsize=(14, 10))
        
        # Get top 5 cities
        top_cities = df['City'].value_counts().head(5).index.tolist()
        
        # Filter data for top cities
        city_data = df[df['City'].isin(top_cities)]
        
        # Create heatmap
        city_means = city_data.groupby('City')[question_cols].mean()
        sns.heatmap(city_means, annot=True, fmt='.1f', cmap='YlOrRd')
        
        plt.title('Average Response by City (Top 5 Cities)')
        plt.savefig(f'{directory}/city_responses.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Response Frequency Heatmap
        plt.figure(figsize=(12, 8))
        
        # Calculate response frequency for each question
        freq_data = {}
        for q in question_cols:
            freq_data[q] = df[q].value_counts().sort_index()
        
        # Convert to dataframe and fill NaN with 0
        freq_df = pd.DataFrame(freq_data).fillna(0)
        
        # Create heatmap
        sns.heatmap(freq_df, annot=True, fmt='.0f', cmap='Blues')
        
        plt.title('Response Frequency Heatmap')
        plt.xlabel('Question')
        plt.ylabel('Response Value')
        plt.savefig(f'{directory}/response_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. Zero Responses Analysis
        plt.figure(figsize=(12, 6))
        
        # Calculate percentage of zero responses for each question
        zero_percentages = []
        for q in question_cols:
            zero_count = (df[q] == 0).sum()
            zero_percentage = (zero_count / len(df)) * 100
            zero_percentages.append(zero_percentage)
        
        # Create bar chart
        plt.bar(question_cols, zero_percentages, color='tomato')
        
        for i, percentage in enumerate(zero_percentages):
            plt.text(i, percentage + 1, f'{percentage:.1f}%', ha='center')
        
        plt.title('Percentage of Zero Responses by Question')
        plt.xlabel('Question')
        plt.ylabel('Percentage of Zero Responses')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.savefig(f'{directory}/zero_responses.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 9. Educational Level vs Response Patterns
        plt.figure(figsize=(15, 10))
        
        # Create a function to create a single subplot
        def plot_edu_response(ax, question, title):
            # Group data by education level
            edu_data = df.groupby('Educational Level')[question].mean().sort_values()
            
            # Create horizontal bar chart
            sns.barplot(x=edu_data.values, y=edu_data.index, ax=ax, palette='Blues')
            
            # Add labels
            ax.set_title(title)
            ax.set_xlabel('Average Response')
            ax.set_xlim(0, 5)
            
            # Add value labels
            for i, v in enumerate(edu_data.values):
                ax.text(v + 0.1, i, f'{v:.1f}', va='center')
        
        # Create subplots for select questions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot for selected questions
        plot_edu_response(axes[0, 0], 'Q1', 'Q1 Response by Education Level')
        plot_edu_response(axes[0, 1], 'Q4', 'Q4 Response by Education Level')
        plot_edu_response(axes[1, 0], 'Q9', 'Q9 Response by Education Level')
        plot_edu_response(axes[1, 1], 'Q10', 'Q10 Response by Education Level')
        
        plt.tight_layout()
        plt.savefig(f'{directory}/education_response.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 10. Employment Status vs Survey Responses
        plt.figure(figsize=(14, 8))
        
        # Calculate mean responses by employment status
        emp_means = df.groupby('Employment Status')[question_cols].mean()
        
        # Create heatmap
        sns.heatmap(emp_means, annot=True, fmt='.1f', cmap='Greens')
        
        plt.title('Average Response by Employment Status')
        plt.savefig(f'{directory}/employment_responses.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a summary image with key findings
        plt.figure(figsize=(14, 8))
        plt.text(0.5, 0.95, 'Survey Analysis: Key Findings', fontsize=24, ha='center', weight='bold')
        
        # Calculate some stats
        response_count = len(df)
        male_count = (df['Gender'] == 'Male').sum()
        female_count = (df['Gender'] == 'Female').sum()
        
        # Most common city
        top_city = df['City'].value_counts().index[0]
        top_city_pct = (df['City'] == top_city).sum() / len(df) * 100
        
        # Average completion time
        avg_time = df['Completion Time'].mean()
        
        # Highest and lowest scoring questions
        q_means = {q: df[q].mean() for q in question_cols}
        highest_q = max(q_means, key=q_means.get)
        highest_q_mean = q_means[highest_q]
        lowest_q = min(q_means, key=q_means.get)
        lowest_q_mean = q_means[lowest_q]
        
        # Create summary text
        summary_text = f"""
        • Total Responses: {response_count}
        • Gender Distribution: {male_count} Male ({male_count/response_count*100:.1f}%), {female_count} Female ({female_count/response_count*100:.1f}%)
        • Most Common City: {top_city} ({top_city_pct:.1f}% of respondents)
        • Average Completion Time: {avg_time:.2f} minutes
        • Highest Scoring Question: {highest_q} (Average: {highest_q_mean:.2f})
        • Lowest Scoring Question: {lowest_q} (Average: {lowest_q_mean:.2f})
        • Most Common Age Group: {df['Age Group'].value_counts().index[0]}
        • Most Common Education Level: {df['Educational Level'].value_counts().index[0]}
        • Most Common Employment Status: {df['Employment Status'].value_counts().index[0]}
        • Most Common Physical Activity Level: {df['Physical Activity Level'].value_counts().index[0]}
        """
        
        plt.text(0.5, 0.5, summary_text, fontsize=14, ha='center', va='center')
        plt.axis('off')
        plt.savefig(f'{directory}/key_findings_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Advanced visualizations created successfully!")
        print(f"10 new visualizations saved to {directory}")

    def create_geographic_heatmap(
            self, 
            data_df: pd.DataFrame,
            directory: str,
            geocode_cache_file: str = 'geocode_cache.json',
            question_to_map: str = None
        ):
        """
        Generate a heatmap visualization on a real-world map based on city names.
        
        Args:
            data_df: DataFrame with the survey data containing a 'City' column
            directory: Directory to save the visualization
            geocode_cache_file: File to cache geocoding results to avoid API limits
            question_to_map: Optional question column to visualize (e.g., 'Q1'). 
                            If None, shows respondent counts.
        
        Returns:
            None - Saves the visualization to the specified directory
        """
        
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Load or create geocode cache
        cache_path = os.path.join(directory, geocode_cache_file)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                geocode_cache = json.load(f)
        else:
            geocode_cache = {}
        
        # Function to geocode a city
        def geocode_city(city_name):
            if city_name in geocode_cache:
                return geocode_cache[city_name]
            
            try:
                # Use Nominatim API for geocoding (free, but rate-limited)
                url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json"
                headers = {'User-Agent': 'SurveyAnalyzer/1.0'}
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    results = response.json()
                    if results:
                        # Take the first result
                        lat = float(results[0]['lat'])
                        lon = float(results[0]['lon'])
                        
                        # Cache the result
                        geocode_cache[city_name] = (lat, lon)
                        
                        # Save the updated cache
                        with open(cache_path, 'w') as f:
                            json.dump(geocode_cache, f)
                        
                        return (lat, lon)
                
                # If we couldn't get coordinates, return None
                return None
            
            except Exception as e:
                print(f"Error geocoding {city_name}: {e}")
                return None
        
        # Clean and count data by city
        city_data = data_df.copy()
        
        # Count respondents by city or aggregate question values
        if question_to_map:
            # Ensure the question column is numeric
            city_data[question_to_map] = pd.to_numeric(city_data[question_to_map], errors='coerce')
            city_metrics = city_data.groupby('City')[question_to_map].agg(['count', 'mean']).reset_index()
            metric_type = 'Average Response'
            value_column = 'mean'
        else:
            # Just count respondents
            city_counts = city_data['City'].value_counts().reset_index()
            city_counts.columns = ['City', 'count']
            city_metrics = city_counts
            metric_type = 'Number of Respondents'
            value_column = 'count'
        
        # Geocode cities
        coordinates = []
        values = []
        city_names = []
        
        print(f"Geocoding cities (this may take a moment)...")
        
        for _, row in city_metrics.iterrows():
            city = row['City']
            coords = geocode_city(city)
            
            if coords:
                coordinates.append(coords)
                values.append(row[value_column])
                city_names.append(city)
        
        print(f"Found coordinates for {len(coordinates)} out of {len(city_metrics)} cities.")
        
        if not coordinates:
            print("No valid coordinates found. Check city names or internet connection.")
            return
        
        # Create the visualization
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1)
        
        # Setup background world map

        try:
            # Create map centered on mean coordinates
            mean_lat = np.mean([lat for lat, _ in coordinates])
            mean_lon = np.mean([lon for _, lon in coordinates])
            
            # Determine appropriate zoom level based on coordinate spread
            lat_range = max([lat for lat, _ in coordinates]) - min([lat for lat, _ in coordinates])
            lon_range = max([lon for _, lon in coordinates]) - min([lon for _, lon in coordinates])
            
            # Add padding
            padding = max(lat_range, lon_range) * 0.5

            min_lat = max(-90, min([lat for lat, _ in coordinates]) - padding)
            max_lat = min(90, max([lat for lat, _ in coordinates]) + padding)
            min_lon = min([lon for _, lon in coordinates]) - padding
            max_lon = max([lon for _, lon in coordinates]) + padding
                        
            # Create map
            m = Basemap(
                projection='merc',
                llcrnrlat=min_lat,
                urcrnrlat=max_lat,
                llcrnrlon=min_lon,
                urcrnrlon=max_lon,
                resolution='i',  # intermediate resolution,
                ax=ax
            )
            
            # Draw map features
            m.drawcoastlines(linewidth=0.5)
            m.drawcountries(linewidth=0.3)
            m.drawstates(linewidth=0.2)
            m.fillcontinents(color='#f5f5f5', lake_color='#e6f2ff')
            m.drawmapboundary(fill_color='#e6f2ff')
            
            # Normalize values for color scaling
            min_val = min(values)
            max_val = max(values)
            normalized_values = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
            
            # Create a custom colormap (blue for higher values)
            cmap = LinearSegmentedColormap.from_list("custom_heatmap", ["#deebf7", "#3182bd"])
            
            # Plot each city as a circle
            for i, ((lat, lon), value, city) in enumerate(zip(coordinates, values, city_names)):
                x, y = m(lon, lat)
                size = 50 + value * 10  # Base size plus scaling
                if size > 1000:  # Cap the maximum size
                    size = 1000
                
                # Plot circle
                m.plot(x, y, 'o', markersize=np.sqrt(size/np.pi), 
                    color=cmap(normalized_values[i]), 
                    alpha=0.7, 
                    markeredgecolor='black', 
                    markeredgewidth=0.5)
                
                # Add label for larger circles
                if value > np.percentile(values, 75):
                    plt.text(x, y, city, fontsize=8, ha='center', va='center', 
                            color='black', fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
            # Create a color bar
            # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_val, max_val))
            # sm.set_array([])  # You need this line when using ScalarMappable directly
            # cbar = plt.colorbar(sm, pad=0.01, shrink=0.75)
            # cbar.set_label(f'{metric_type}')
            # Create a proper mappable object that's connected to an actual plot
            scatter = plt.scatter([], [], c=[], cmap=cmap, norm=plt.Normalize(min(values), max(values)))
            scatter.set_array(np.array(values))

            # Now create the colorbar using this proper mappable
            cbar = plt.colorbar(scatter, pad=0.01, shrink=0.75)
            cbar.set_label(f'{metric_type}')
                        
            # Set title
            if question_to_map:
                title = f"Geographic Distribution of {question_to_map} Responses"
            else:
                title = "Geographic Distribution of Survey Respondents"
            
            plt.title(title, fontsize=14)
            
        except ImportError:
            # If basemap is not available, create a scatter plot instead
            plt.scatter([lon for _, lon in coordinates], 
                    [lat for lat, _ in coordinates], 
                    c=values, 
                    cmap='Blues', 
                    s=[v*20 for v in values], 
                    alpha=0.7, 
                    edgecolor='black')
            
            # Add labels for points with higher values
            for i, ((lat, lon), value, city) in enumerate(zip(coordinates, values, city_names)):
                if value > np.percentile(values, 75):
                    plt.text(lon, lat, city, fontsize=8, ha='center', va='center')
            
            plt.colorbar(label=metric_type)
            
            if question_to_map:
                title = f"Geographic Distribution of {question_to_map} Responses (Simple Plot)"
            else:
                title = "Geographic Distribution of Survey Respondents (Simple Plot)"
            
            plt.title(title, fontsize=14)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
        
        # Save the figure
        filename = f'geographic_heatmap{"_" + question_to_map if question_to_map else ""}.png'
        plt.savefig(f'{directory}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Geographic heatmap saved as {filename}")
        
        # Also save the geocoding cache in case it was updated
        with open(cache_path, 'w') as f:
            json.dump(geocode_cache, f)
        
        return None
    

class FishersExactAnalyzer:
    """
    Performs Fisher's Exact test for independence and provides enhanced interpretability
    through visualizations, hypothesis testing, and additional metrics like Odds Ratio.
    """

    @staticmethod
    def analyze(data, col1, col2):
        # Create contingency table
        contingency_table = pd.crosstab(data[col1], data[col2])

        print(contingency_table.shape)

        # Check if the table is 2x2 for Fisher's Exact Test
        if contingency_table.shape != (2, 2):
            raise ValueError("Fisher's Exact Test is only applicable to 2x2 tables.")

        # Perform Fisher's Exact test
        _, p_value = stats.fisher_exact(contingency_table)

        # Calculate Odds Ratio and its 95% Confidence Interval
        odds_ratio, ci_lower, ci_upper = FishersExactAnalyzer.odds_ratio_and_ci(contingency_table)

        # Return Fisher's Exact test results
        return {
            "p-value"              : p_value,
            "Odds Ratio"           : odds_ratio,
            "Confidence Interval"  : (ci_lower, ci_upper),
            "Contingency Table"    : contingency_table
        }

    @staticmethod
    def odds_ratio_and_ci(contingency_table):
        """
        Calculate the Odds Ratio (OR) and its 95% Confidence Interval.
        """
        # Extract the counts from the contingency table
        a, b, c, d = contingency_table.values.flatten()

        # Calculate Odds Ratio
        odds_ratio = (a * d) / (b * c)
        
        # Calculate the 95% Confidence Interval for the Odds Ratio
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)  # Standard error of log(OR)
        log_or = np.log(odds_ratio)  # Log of the odds ratio
        z = 1.96  # For 95% confidence interval
        lower = np.exp(log_or - z * se_log_or)
        upper = np.exp(log_or + z * se_log_or)
        
        return odds_ratio, lower, upper

    @staticmethod
    def plot_contingency_table(data, col1, col2):
        """
        Visualize the contingency table as a heatmap for better understanding.
        """
        contingency_table = pd.crosstab(data[col1], data[col2])
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Contingency Table Heatmap: {col1} vs {col2}')
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.show()
        return contingency_table

    @staticmethod
    def interpret_results(p_value, significance_level=0.05):
        """
        Interpret the p-value in the context of hypothesis testing.

        Args:
            p_value: The p-value from Fisher's Exact test.
            significance_level: The threshold below which the result is considered statistically significant.
        
        Returns:
            Interpretation message.
        """
        if p_value < significance_level:
            return f"The test is significant (p-value = {p_value:.4f}). Reject the null hypothesis: There is a significant association between the variables."
        else:
            return f"The test is not significant (p-value = {p_value:.4f}). Fail to reject the null hypothesis: No significant association between the variables."

    @staticmethod
    def report(data, col1, col2, significance_level=0.05):
        """
        Run Fisher's Exact test, interpret the results, and generate visualizations.
        """
        results = FishersExactAnalyzer.analyze(data, col1, col2)
        p_value = results["p-value"]
        odds_ratio = results["Odds Ratio"]
        ci_lower, ci_upper = results["Confidence Interval"]
        
        # Begin fancy report
        print("="*50)
        print(f" Fisher's Exact Test Report ({col1}, {col2}) ".center(50, "="))
        print("="*50)
        print(f"**p-value**: {p_value:.4f}")
        print(f"**Odds Ratio**: {odds_ratio:.4f}")
        print(f"**95% Confidence Interval**: ({ci_lower:.4f}, {ci_upper:.4f})")
        print("\nInterpretation:")
        print(FishersExactAnalyzer.interpret_results(p_value, significance_level))
        print("-"*50)
        
        # Visualize the contingency table with a fancy plot
        contingency_table = FishersExactAnalyzer.plot_contingency_table(data, col1, col2)
        print("\n**Contingency Table Visualization**:")
        print(contingency_table)
        plt.title(f"Contingency Table: {col1} vs {col2}", fontsize=14)
        plt.tight_layout()
        plt.show()
        print("="*50)

        results = {
            "p_value"            : float(p_value),
            "odds_ratio"         : float(odds_ratio),
            "confidence_interval": (ci_lower, ci_upper),
            "contingency_table"  : contingency_table
        }

        return results

class ChiSquareAnalyzer:
    """
    Performs Chi-Square test for independence and provides enhanced interpretability
    through visualizations, hypothesis testing, and p-value interpretation.

    Example
    -------
    Test whether eating habits (e.g., frequency of consuming sugary beverages) depend on
    age group or physical activity level.
    """

    @staticmethod
    def analyze(data, col1, col2):
        # Create contingency table
        contingency_table = pd.crosstab(data[col1], data[col2])

        # sample_size_effect
        total_samples          = contingency_table.sum().sum()
        expected_frequencies   = stats.chi2_contingency(contingency_table)[3]
        min_expected_frequency = np.min(expected_frequencies)
        
        # Perform Chi-Square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        # Effect Size
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (total_samples * min_dim))

        # Return Chi-Square test results
        return {
            "Chi2"                 : chi2,
            "p-value"              : p,
            "Degrees of Freedom"   : dof,
            #"Expected Frequencies" : expected,
            #"Contingency Table"    : contingency_table,
            "Sample Size Effect"   : min_expected_frequency,
            "Effect Size"          : cramers_v,
        }

    @staticmethod
    def plot_contingency_table(data, col1, col2):
        """
        Visualize the contingency table as a heatmap for better understanding.
        """
        contingency_table = pd.crosstab(data[col1], data[col2])
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Contingency Table Heatmap: {col1} vs {col2}')
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.show()
        return contingency_table

    @staticmethod
    def interpret_results(p_value, significance_level=0.05):
        """
        Interpret the p-value in the context of hypothesis testing.

        Args:
            p_value: The p-value from the Chi-Square test.
            significance_level: The threshold below which the result is considered statistically significant.
        
        Returns:
            Interpretation message.
        """
        if p_value < significance_level:
            return f"The test is significant (p-value = {p_value:.4f}). Reject the null hypothesis: There is a significant association between the variables."
        else:
            return f"The test is not significant (p-value = {p_value:.4f}). Fail to reject the null hypothesis: No significant association between the variables."

    @staticmethod
    def sample_size_effect(data, col1, col2):
        """
        Check if sample size is sufficient for the Chi-Square test. Small sample sizes
        might lead to inaccurate results.
        """
        contingency_table      = pd.crosstab(data[col1], data[col2])
        total_samples          = contingency_table.sum().sum()
        expected_frequencies   = stats.chi2_contingency(contingency_table)[3]
        min_expected_frequency = np.min(expected_frequencies)
        
        # Rule of thumb: Expected frequencies should be at least 5
        if min_expected_frequency < 5:
            print(f"Warning: The expected frequency for some cells is below 5, which may lead to inaccurate results. Sample size (total = {total_samples}) may be too small.")
        else:
            print(f"Sample size appears sufficient. Total samples: {total_samples}")
        return min_expected_frequency

    @staticmethod
    def effect_size(data, col1, col2):
        """
        Calculate Cramér's V effect size for the Chi-Square test.
        
        Cramér's V is used to measure the strength of association between two categorical variables.
        """
        contingency_table = pd.crosstab(data[col1], data[col2])
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        print(f"Cramér's V: {cramers_v:.4f} (0 = no association, 1 = strong association)")
        return cramers_v
    
    @staticmethod
    def report(data, col1, col2, significance_level=0.05, expanded : bool = False):
        """
        Run a Chi-Square test, interpret the results, and generate visualizations.
        """
        results = ChiSquareAnalyzer.analyze(data, col1, col2)
        chi2, p_value = results["Chi2"], results["p-value"]
        
        # Begin fancy report
        print("="*50)
        print(f" Chi-Square Test Report ({col1}, {col2}) ".center(50, "="))
        print("="*50)
        print(f"**Chi-Square Statistic**: {chi2:.4f}")
        print(f"**p-value**: {p_value:.4f}")
        print("\nInterpretation:")
        print(ChiSquareAnalyzer.interpret_results(p_value, significance_level))
        print("-"*50)
        
        # Display effect size
        effect_size = ChiSquareAnalyzer.effect_size(data, col1, col2)
        print("\n**Effect Size**:")
        print(effect_size)
        print("-"*50)
        
        # Display sample size considerations
        sample_size_effect = ChiSquareAnalyzer.sample_size_effect(data, col1, col2)
        print("\n**Sample Size Considerations**:")
        print(sample_size_effect)
        print("-"*50)
        
        # Visualize the contingency table with a fancy plot
        contingency_table = ChiSquareAnalyzer.plot_contingency_table(data, col1, col2)
        print("\n**Contingency Table Visualization**:")
        print(contingency_table)
        plt.title(f"Contingency Table: {col1} vs {col2}", fontsize=14)
        plt.tight_layout()
        plt.show()
        print("="*50)

        results = {
            "p_value"            : float(p_value),
            "effect_size"        : float(effect_size),
            "sample_size_effect" : float(sample_size_effect), 
            "contingency_table"  : contingency_table
        }

        return results


class ANOVAAnalyzer:
    """
    Performs One-Way ANOVA test and provides enhanced interpretability
    through visualizations, hypothesis testing interpretation, effect size,
    and sample size considerations.
    
    Example
    -------
    Test whether average exam scores differ among different teaching methods.
    """

    @staticmethod
    def analyze(data, dependent_var, group_var):
        """
        Perform a One-Way ANOVA test to compare means across multiple groups.
        Also returns effect size and minimum group size.

        Args:
            data (pd.DataFrame): The dataset.
            dependent_var (str): The numerical variable (dependent variable).
            group_var (str): The categorical variable (grouping factor).

        Returns:
            dict: Dictionary containing ANOVA statistics and additional information.
        """
        # Group data by the categorical variable
        groups = [group[dependent_var].dropna() for _, group in data.groupby(group_var)]
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Calculate degrees of freedom
        df_between = len(groups) - 1
        df_within = len(data) - len(groups)
        
        # Sum of squares
        overall_mean = data[dependent_var].mean()
        ss_between = sum(len(group) * (group.mean() - overall_mean) ** 2 for group in groups)
        ss_within = sum(((group - group.mean()) ** 2).sum() for group in groups)
        ss_total = ss_between + ss_within

        # Mean squares
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        # Calculate the effect size (eta-squared)
        eta_squared = ss_between / ss_total

        # Calculate the minimum group size
        group_sizes = data.groupby(group_var)[dependent_var].count()
        min_group_size = group_sizes.min()

        # Return all relevant statistics
        return {
            "F-statistic": f_stat,
            "p-value": p_value,
            "eta_squared": eta_squared,
            "min_group_size": min_group_size
        }

    @staticmethod
    def plot_data_distribution(data, dependent_var, group_var):
        """
        Visualize the data distribution using a boxplot and violin plot.

        Args:
            data (pd.DataFrame): The dataset.
            dependent_var (str): The numerical variable (dependent variable).
            group_var (str): The categorical variable (grouping factor).
        """
        plt.figure(figsize=(10, 5))
        
        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(x=group_var, y=dependent_var, data=data, palette="Set2")
        plt.title(f'Boxplot of {dependent_var} by {group_var}')
        plt.xticks(rotation=45)
        
        # Violin plot
        plt.subplot(1, 2, 2)
        sns.violinplot(x=group_var, y=dependent_var, data=data, palette="Set2", inner="quartile")
        plt.title(f'Violin Plot of {dependent_var} by {group_var}')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def interpret_results(p_value, significance_level=0.05):
        """
        Interpret the p-value in the context of hypothesis testing.

        Args:
            p_value (float): The p-value from the ANOVA test.
            significance_level (float, optional): The threshold for significance. Default is 0.05.

        Returns:
            str: Interpretation of the hypothesis test results.
        """
        if p_value < significance_level:
            return f"The test is significant (p-value = {p_value:.4f}). Reject the null hypothesis: At least one group mean is significantly different."
        else:
            return f"The test is not significant (p-value = {p_value:.4f}). Fail to reject the null hypothesis: No significant difference among group means."

    @staticmethod
    def sample_size_effect(data, dependent_var, group_var):
        """
        Assess whether the sample size is sufficient for ANOVA.

        Args:
            data (pd.DataFrame): The dataset.
            dependent_var (str): The numerical variable (dependent variable).
            group_var (str): The categorical variable (grouping factor).

        Returns:
            str: Message about sample size adequacy.
        """
        group_sizes = data.groupby(group_var)[dependent_var].count()
        min_group_size = group_sizes.min()

        if min_group_size < 10:
            print(f"Warning: Some groups have fewer than 10 samples. ANOVA results may be unreliable.")
            return min_group_size
        else:
            print(f"Sample sizes are sufficient. Smallest group size: {min_group_size}.")
            return min_group_size

    @staticmethod
    def effect_size(data, dependent_var, group_var):
        """
        Calculate eta-squared (η²), a measure of effect size for ANOVA.

        Args:
            data (pd.DataFrame): The dataset.
            dependent_var (str): The numerical variable (dependent variable).
            group_var (str): The categorical variable (grouping factor).

        Returns:
            str: Effect size interpretation.
        """
        groups = [group[dependent_var].dropna() for _, group in data.groupby(group_var)]
        overall_mean = data[dependent_var].mean()
        ss_between = sum(len(group) * (group.mean() - overall_mean) ** 2 for group in groups)
        ss_total = sum((data[dependent_var] - overall_mean) ** 2)
        eta_squared = ss_between / ss_total

        print(f"Effect size (η²) = {eta_squared:.4f} (Small: 0.01, Medium: 0.06, Large: 0.14)")
        return eta_squared

    @staticmethod
    def report(data, dependent_var, group_var, significance_level=0.05):
        """
        Run ANOVA, interpret results, and generate visualizations.

        Args:
            data (pd.DataFrame): The dataset.
            dependent_var (str): The numerical variable (dependent variable).
            group_var (str): The categorical variable (grouping factor).
            significance_level (float, optional): The threshold for significance. Default is 0.05.
        """
        results = ANOVAAnalyzer.analyze(data, dependent_var, group_var)
        f_stat, p_value = results["F-statistic"], results["p-value"]

        # Print results
        print(f"F-Statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(ANOVAAnalyzer.interpret_results(p_value, significance_level))
        
        # Display effect size
        print(ANOVAAnalyzer.effect_size(data, dependent_var, group_var))
        
        # Display sample size considerations
        print(ANOVAAnalyzer.sample_size_effect(data, dependent_var, group_var))
        
        # Visualize the data
        ANOVAAnalyzer.plot_data_distribution(data, dependent_var, group_var)


class NonParametricTests:
    """
    Performs Mann-Whitney U Test and Kruskal-Wallis Test with additional interpretability,
    including hypothesis testing interpretation, effect size calculations, sample size checks, and visualizations.
    
    Example
    -------
    Test whether two independent groups have different distributions (Mann-Whitney U Test),
    or whether more than two independent groups differ (Kruskal-Wallis Test).
    """

    @staticmethod
    def mann_whitney(data, col, group):
        """
        Perform Mann-Whitney U test for two independent groups.

        Args:
            data (pd.DataFrame): The dataset.
            col (str): The numerical variable to compare.
            group (str): The categorical variable defining the two groups.

        Returns:
            dict: U-statistic and p-value.
        """
        unique_groups = data[group].unique()
        if len(unique_groups) != 2:
            raise ValueError("Mann-Whitney U Test requires exactly two groups.")

        group1 = data[data[group] == unique_groups[0]][col].dropna()
        group2 = data[data[group] == unique_groups[1]][col].dropna()

        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        return {
            "U-statistic": u_stat,
            "p-value": p_value
        }

    @staticmethod
    def kruskal_wallis(data, col, group):
        """
        Perform Kruskal-Wallis test for multiple independent groups.

        Args:
            data (pd.DataFrame): The dataset.
            col (str): The numerical variable to compare.
            group (str): The categorical variable defining the groups.

        Returns:
            dict: H-statistic and p-value.
        """
        groups = [g[col].dropna() for _, g in data.groupby(group)]
        h_stat, p_value = stats.kruskal(*groups)

        return {
            "H-statistic": h_stat,
            "p-value": p_value
        }

    @staticmethod
    def interpret_results(test_name, p_value, significance_level=0.05):
        """
        Interpret the p-value for hypothesis testing.

        Args:
            test_name (str): The name of the statistical test.
            p_value (float): The computed p-value.
            significance_level (float, optional): The significance threshold. Default is 0.05.

        Returns:
            str: Interpretation of the hypothesis test result.
        """
        if p_value < significance_level:
            return f"{test_name} is significant (p-value = {p_value:.4f}). Reject the null hypothesis: There is a significant difference between groups."
        else:
            return f"{test_name} is not significant (p-value = {p_value:.4f}). Fail to reject the null hypothesis: No significant difference between groups."

    @staticmethod
    def plot_data_distribution(data, col, group):
        """
        Visualize data distribution using boxplots and histograms.

        Args:
            data (pd.DataFrame): The dataset.
            col (str): The numerical variable.
            group (str): The categorical variable.
        """
        plt.figure(figsize=(10, 5))

        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(x=group, y=col, data=data, palette="Set2")
        plt.title(f'Boxplot of {col} by {group}')
        plt.xticks(rotation=45)

        # Histogram
        plt.subplot(1, 2, 2)
        sns.histplot(data, x=col, hue=group, element="step", common_norm=False, kde=True)
        plt.title(f'Distribution of {col} by {group}')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def effect_size_mann_whitney(u_stat, n1, n2):
        """
        Compute effect size (r) for the Mann-Whitney U test.

        Args:
            u_stat (float): The U-statistic.
            n1 (int): Sample size of group 1.
            n2 (int): Sample size of group 2.

        Returns:
            str: Effect size interpretation.
        """
        r = u_stat / (n1 * n2)
        return f"Effect size (r) = {r:.4f} (Small: 0.1, Medium: 0.3, Large: 0.5)"

    @staticmethod
    def effect_size_kruskal(h_stat, n):
        """
        Compute effect size (eta-squared) for the Kruskal-Wallis test.

        Args:
            h_stat (float): The H-statistic.
            n (int): Total sample size.

        Returns:
            str: Effect size interpretation.
        """
        eta_squared = (h_stat - len(n)) / (n - 1)
        return f"Effect size (η²) = {eta_squared:.4f} (Small: 0.01, Medium: 0.06, Large: 0.14)"

    @staticmethod
    def sample_size_check(data, col, group):
        """
        Check whether the sample size is sufficient for non-parametric tests.

        Args:
            data (pd.DataFrame): The dataset.
            col (str): The numerical variable.
            group (str): The categorical variable.

        Returns:
            str: Message about sample size adequacy.
        """
        group_sizes = data.groupby(group)[col].count()
        min_group_size = group_sizes.min()

        if min_group_size < 10:
            return f"Warning: Some groups have fewer than 10 samples. Non-parametric results may be unreliable."
        else:
            return f"Sample sizes are sufficient. Smallest group size: {min_group_size}."

    @staticmethod
    def report_mann_whitney(data, col, group, significance_level=0.05):
        """
        Perform Mann-Whitney U test, interpret results, and generate visualizations.

        Args:
            data (pd.DataFrame): The dataset.
            col (str): The numerical variable.
            group (str): The categorical variable.
            significance_level (float, optional): The threshold for significance. Default is 0.05.
        """
        results = NonParametricTests.mann_whitney(data, col, group)
        u_stat, p_value = results["U-statistic"], results["p-value"]

        print(f"U-Statistic: {u_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(NonParametricTests.interpret_results("Mann-Whitney U Test", p_value, significance_level))

        group_sizes = data.groupby(group)[col].count()
        print(NonParametricTests.effect_size_mann_whitney(u_stat, *group_sizes))
        print(NonParametricTests.sample_size_check(data, col, group))

        NonParametricTests.plot_data_distribution(data, col, group)

    @staticmethod
    def report_kruskal_wallis(data, col, group, significance_level=0.05):
        """
        Perform Kruskal-Wallis test, interpret results, and generate visualizations.

        Args:
            data (pd.DataFrame): The dataset.
            col (str): The numerical variable.
            group (str): The categorical variable.
            significance_level (float, optional): The threshold for significance. Default is 0.05.
        """
        results = NonParametricTests.kruskal_wallis(data, col, group)
        h_stat, p_value = results["H-statistic"], results["p-value"]

        print(f"H-Statistic: {h_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(NonParametricTests.interpret_results("Kruskal-Wallis Test", p_value, significance_level))

        print(NonParametricTests.effect_size_kruskal(h_stat, len(data)))
        print(NonParametricTests.sample_size_check(data, col, group))

        NonParametricTests.plot_data_distribution(data, col, group)

class CorrelationAnalyzer:
    """Performs Spearman's Rank Correlation test."""
    @staticmethod
    def analyze(data, col1, col2):
        corr, p_value = stats.spearmanr(data[col1], data[col2])
        return {"Correlation": corr, "p-value": p_value}

class FactorAnalysisAnalyzer:
    """Performs Factor Analysis to identify latent variables."""
    @staticmethod
    def analyze(data, n_factors):
        fa = FactorAnalysis(n_components=n_factors)
        fa.fit(data)
        return {"Components": fa.components_}

class MANOVAAnalyzer:
    """Performs Multivariate Analysis of Variance (MANOVA)."""
    @staticmethod
    def analyze(data, dependent_vars, independent_var):
        formula = " + ".join(dependent_vars) + " ~ " + independent_var
        manova = MANOVA.from_formula(formula, data)
        return manova.mv_test()


class LogisticRegressionAnalyzer:
    """Performs Logistic Regression analysis."""
    @staticmethod
    def analyze(data, dependent_var, independent_vars):
        model = LogisticRegression()
        model.fit(data[independent_vars], data[dependent_var])
        return {"Coefficients": model.coef_, "Intercept": model.intercept_}

class ClusterAnalysis:
    """Performs Cluster Analysis using K-Means."""
    @staticmethod
    def analyze(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        return {"Labels": kmeans.labels_}
