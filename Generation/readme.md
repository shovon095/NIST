
This repository contains a Python script that processes a set of articles, generates summaries using OpenAI's GPT models, and attempts to enhance these summaries 
to mimic human writing styles. The script is designed to handle large datasets, generate context-aware summaries, and iteratively refine them to make them appear 
more human-like. Additionally, the script performs tasks such as adjusting tone and style based on user preferences, and simulating different personas to evaluate 
the "human-ness" of the generated summaries.

Features:

Text Summarization: Generates summaries of large text datasets using OpenAI's GPT models.
Human-like Refinement: Enhances the generated summaries by adjusting tone, style, and sentence structure to mimic human writing.
Context-Aware Summarization: Automatically adjusts the summarization process based on the detected context of the text (e.g., technical, narrative).
Simulated Personas: Uses simulated personas to evaluate whether the summaries are detected as AI-generated or human-written.
Batch Processing: Supports batch processing of multiple articles, with summaries saved in XML format and results logged in CSV format.
Customization: Allows customization of the summary based on detail level, focus area, tone, and style.

Requirements:

Python 3.6 or higher

openai;
os;
random;
csv;
xml.etree.ElementTree;
concurrent.futures;
matplotlib;

Set Up OpenAI API Key:

Before running the script, ensure that you have your OpenAI API key set up. Replace 'YOUR OPENAI API KEY' in the script with your actual API key.

Usage:

input_directory: The directory where the input article files are stored.
output_directory: The directory where the generated summaries will be saved.
results_file: The CSV file path where the GPT-3.5 results will be saved.
topics_file: The SGML file that maps articles to specific topics.


Functionality Overview:

process_and_save_summaries: The main function that orchestrates the summarization process. It processes all articles by topics, generates summaries, refines them to appear more human-like, and saves the summaries in XML format and results in CSV format.
summarize_articles: Summarizes multiple articles and blends the summaries to produce a diverse final output.
context_aware_summarization: Analyzes the context of the text and adjusts the summarization strategy accordingly.
add_human_like_features: Enhances the generated summary by introducing human-like elements such as varied sentence structures and tone adjustments.
check_with_simulated_personas: Uses different personas to detect if the summary is more likely AI-generated or human-written.
iterative_summary_refinement: Refines the summary iteratively to make it more human-like.

Example Input:

Article Directory: The script expects articles to be placed in the specified input_directory.
Topics File: An SGML file (topics_file) that lists the topics and associated article filenames. This file helps map articles to their respective topics.

Output:

XML Summaries: The script generates an XML file containing the refined summaries for each topic.
CSV Results: A CSV file containing the detection results (AI or Human) for each topic's summary.


Example Directory Structure:

/path/to/your/project/
│
├── script_name.py
├── input_directory/
│   ├── article1.txt
│   ├── article2.txt
│   └── ...
├── output_directory/
│   ├── summaries.xml
│   └── ...
├── topics_file.sgml
└── gpt_35_results.csv

Customization:

The script is highly customizable. You can modify:

User Profiles: Define different user profiles with specific tone and style preferences.

Summarization Parameters: Adjust parameters like MAX_TOKENS_PER_REQUEST and MAX_WORDS_FINAL_SUMMARY to control the size and granularity of summaries.

