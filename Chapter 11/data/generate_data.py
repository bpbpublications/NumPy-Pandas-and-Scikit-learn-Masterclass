# Usage: python generate_data.py --n_samples 500 --edu_min 10 --edu_max 22 --exp_min 1 --exp_max 35 --hours_min 30 --hours_max 50 --age_min 20 --age_max 60 --location_max 10


import numpy as np
import pandas as pd
import argparse 

# Argument Parser for dataset generation
parser = argparse.ArgumentParser(description='Generate synthetic income dataset')
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to generate')
parser.add_argument('--edu_min', type=int, default=12, help='Minimum education level')
parser.add_argument('--edu_max', type=int, default=20, help='Maximum education level')
parser.add_argument('--exp_min', type=int, default=0, help='Minimum years of experience')
parser.add_argument('--exp_max', type=int, default=40, help='Maximum years of experience')
parser.add_argument('--hours_min', type=int, default=20, help='Minimum work hours per week')
parser.add_argument('--hours_max', type=int, default=60, help='Maximum work hours per week')
parser.add_argument('--age_min', type=int, default=18, help='Minimum age')
parser.add_argument('--age_max', type=int, default=65, help='Maximum age')
parser.add_argument('--location_max', type=int, default=5, help='Number of location categories')
args = parser.parse_args()

# Generate Synthetic Data
np.random.seed(42)
n_samples = args.n_samples

data = pd.DataFrame({
    'education_level': np.random.randint(args.edu_min, args.edu_max + 1, size=n_samples),
    'experience_years': np.random.randint(args.exp_min, args.exp_max, size=n_samples),
    'hours_per_week': np.random.randint(args.hours_min, args.hours_max, size=n_samples),
    'age': np.random.randint(args.age_min, args.age_max, size=n_samples),
    'gender': np.random.choice([0, 1], size=n_samples),
    'marital_status': np.random.choice([0, 1], size=n_samples),
    'location_index': np.random.randint(0, args.location_max, size=n_samples)
})

data['income'] = (
    data['education_level'] * 2000 +
    data['experience_years'] * 1500 +
    data['hours_per_week'] * 100 +
    data['age'] * 80 +
    data['gender'] * 1000 +
    data['marital_status'] * 500 +
    data['location_index'] * 700 +
    np.random.normal(0, 5000, size=n_samples)
)

data.to_csv('synthetic_income_data.csv', index=False)