AI-Powered Ticket Creation and Categorisation System
 Project Overview:

The AI-Powered Ticket Creation and Categorisation System is an intelligent application designed to automate the creation, categorisation, and prioritisation of IT support tickets based on user input.
Using Natural Language Processing (NLP) and Machine Learning techniques, the system analyzes ticket descriptions and predicts both the issue category and priority level, helping organizations streamline support workflows and reduce manual effort.

 Problem Statement:

In traditional IT support systems, tickets are manually categorized and prioritized, which is time-consuming, error-prone, and inefficient.
This project aims to automate this process using AI, enabling faster ticket handling and improved service response times.

 Objectives:

Automate ticket creation from raw user input

Predict the category of the issue using NLP-based ML models

Predict the priority of the ticket to indicate urgency

Provide a backend API and user-friendly interface

Demonstrate real-world AI application in IT service management

 Solution Approach:

Text preprocessing using NLP techniques

Feature extraction using TF-IDF

Classification using Random Forest models

Modular Python codebase for scalability

Web-based interface for real-time predictions

 Technologies Used:

Programming Language: Python

Machine Learning: Scikit-learn

NLP: TF-IDF Vectorization

Backend: Flask

Frontend: HTML, CSS, JavaScript

Model Persistence: Joblib

Deployment: Railway / Localhost

Development Environment: VS Code

Project Structure:
AI Ticket Generation/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── rf_category_model.pkl
│   ├── category_vectorizer.pkl
│   ├── category_label_encoder.pkl
│   ├── rf_priority_model.pkl
│   ├── priority_vectorizer.pkl
│   └── priority_label_encoder.pkl
│
├── scripts/
│   ├── __init__.py
│   ├── clean_text.py
│   ├── preprocess.py
│   ├── make_splits.py
│   ├── train_model.py
│   ├── predict.py
│   └── generate_ticket.py
│
├── templates/
│   └── index.html
│
└── data/
    └── IT Support Ticket Data.csv

Dataset Description

Contains over 25,000+ IT support tickets

Includes fields such as:

Ticket Description

Category

Priority

Data cleaning and preprocessing performed to remove noise and inconsistencies

 Data Preprocessing:

Lowercasing text

Removing URLs, emails, and special characters

Removing extra whitespaces

TF-IDF vectorization for feature extraction

 Model Used:
✔ Random Forest Classifier

Selected after comparing multiple ML models

Provides good accuracy and robustness

Separate models trained for:

Category prediction

Priority prediction

System Architecture:
User Input
    ↓
Frontend (HTML UI)
    ↓
Flask Backend
    ↓
Text Preprocessing
    ↓
Random Forest Models
    ↓
Ticket Creation (Category + Priority)

Implementation Steps

Load and preprocess dataset

Train Random Forest models for category and priority

Save trained models using Joblib

Create Flask API to load models and handle predictions

Build frontend interface for user input

Integrate ticket creation logic

Deploy application

 Results:

Automated ticket categorisation and prioritisation

Real-time predictions from user input

Significant reduction in manual intervention

Accurate and consistent classification results

 Output:

Ticket ID generation

Predicted Category

Predicted Priority

Ticket status and timestamp

 Challenges Faced:

Class imbalance in dataset

Handling long and noisy text inputs

Model deployment limitations on free platforms

Managing Python package imports in modular structure

 Future Improvements:

Integrate deep learning models (BiLSTM, BERT)

Add ticket storage using database (MySQL / SQLite)

Implement user authentication

Add confidence scores for predictions

Build analytics dashboard for ticket trends

 Conclusion:

This project successfully demonstrates how AI and NLP can be applied to automate IT support ticket management. By leveraging machine learning models and a modular backend architecture, the system provides an efficient, scalable, and intelligent solution for real-world service management challenges.

Author:

Sourav Pati
