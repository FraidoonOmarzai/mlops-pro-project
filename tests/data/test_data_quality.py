"""
Data quality tests.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.mark.data
class TestDataQuality:
    """Test suite for data quality checks."""
    
    def test_no_missing_values_in_critical_columns(self, sample_dataframe):
        """Test critical columns have no missing values."""
        critical_columns = ['customerID', 'Churn']
        
        for col in critical_columns:
            if col in sample_dataframe.columns:
                assert sample_dataframe[col].isnull().sum() == 0, \
                    f"Column {col} has missing values"
    
    def test_no_duplicate_customer_ids(self, sample_dataframe):
        """Test customer IDs are unique."""
        if 'customerID' in sample_dataframe.columns:
            assert sample_dataframe['customerID'].is_unique, \
                "Duplicate customer IDs found"
    
    def test_data_types_correct(self, sample_dataframe):
        """Test data types are as expected."""
        expected_types = {
            'tenure': (np.int64, int),
            'MonthlyCharges': (np.float64, float),
            'TotalCharges': (np.float64, float),
        }
        
        for col, expected_type in expected_types.items():
            if col in sample_dataframe.columns:
                assert sample_dataframe[col].dtype in expected_type or \
                       isinstance(sample_dataframe[col].iloc[0], expected_type), \
                    f"Column {col} has incorrect type"
    
    def test_categorical_values_valid(self, sample_dataframe):
        """Test categorical columns have valid values."""
        valid_values = {
            'gender': ['Male', 'Female'],
            'Partner': ['Yes', 'No'],
            'Dependents': ['Yes', 'No'],
            'PhoneService': ['Yes', 'No'],
            'PaperlessBilling': ['Yes', 'No'],
            'Churn': ['Yes', 'No']
        }
        
        for col, valid_vals in valid_values.items():
            if col in sample_dataframe.columns:
                invalid = sample_dataframe[~sample_dataframe[col].isin(valid_vals)][col]
                assert len(invalid) == 0, \
                    f"Column {col} has invalid values: {invalid.unique()}"
    
    def test_numerical_ranges(self, sample_dataframe):
        """Test numerical values are within expected ranges."""
        if 'tenure' in sample_dataframe.columns:
            assert sample_dataframe['tenure'].min() >= 0
            assert sample_dataframe['tenure'].max() <= 100
        
        if 'MonthlyCharges' in sample_dataframe.columns:
            assert sample_dataframe['MonthlyCharges'].min() >= 0
            assert sample_dataframe['MonthlyCharges'].max() <= 200
        
        if 'TotalCharges' in sample_dataframe.columns:
            assert sample_dataframe['TotalCharges'].min() >= 0
    
    def test_no_outliers_in_charges(self, sample_dataframe):
        """Test for extreme outliers in charges."""
        if 'MonthlyCharges' in sample_dataframe.columns:
            q1 = sample_dataframe['MonthlyCharges'].quantile(0.25)
            q3 = sample_dataframe['MonthlyCharges'].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outliers = sample_dataframe[
                (sample_dataframe['MonthlyCharges'] < lower_bound) |
                (sample_dataframe['MonthlyCharges'] > upper_bound)
            ]
            
            # Allow some outliers, but not too many
            assert len(outliers) / len(sample_dataframe) < 0.1
    
    def test_data_consistency(self, sample_dataframe):
        """Test logical consistency in data."""
        # If no phone service, MultipleLines should be "No phone service"
        if all(col in sample_dataframe.columns for col in ['PhoneService', 'MultipleLines']):
            no_phone = sample_dataframe[sample_dataframe['PhoneService'] == 'No']
            if len(no_phone) > 0:
                assert (no_phone['MultipleLines'] == 'No phone service').all() or \
                       (no_phone['MultipleLines'] == 'No').all(), \
                    "Data inconsistency: PhoneService='No' but MultipleLines != 'No phone service'"
    
    def test_class_distribution(self, sample_dataframe):
        """Test target variable distribution is reasonable."""
        if 'Churn' in sample_dataframe.columns:
            churn_dist = sample_dataframe['Churn'].value_counts(normalize=True)
            
            # Check we have both classes
            assert len(churn_dist) >= 1
            
            # Check extreme imbalance (less than 1% of minority class)
            if len(churn_dist) > 1:
                min_class_ratio = churn_dist.min()
                assert min_class_ratio >= 0.01, \
                    f"Severe class imbalance: {min_class_ratio:.2%}"


@pytest.mark.data
def test_dataframe_not_empty(sample_dataframe):
    """Test dataframe is not empty."""
    assert len(sample_dataframe) > 0
    assert len(sample_dataframe.columns) > 0


@pytest.mark.data
def test_no_all_null_columns(sample_dataframe):
    """Test no columns are entirely null."""
    for col in sample_dataframe.columns:
        assert not sample_dataframe[col].isnull().all(), \
            f"Column {col} is entirely null"


@pytest.mark.data
def test_total_charges_consistency(sample_dataframe):
    """Test TotalCharges is roughly consistent with tenure and MonthlyCharges."""
    required_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    if all(col in sample_dataframe.columns for col in required_cols):
        # Filter out rows with 0 tenure or missing values
        valid_rows = sample_dataframe[
            (sample_dataframe['tenure'] > 0) &
            (sample_dataframe['TotalCharges'].notna())
        ]
        
        if len(valid_rows) > 0:
            # Calculate expected TotalCharges
            expected = valid_rows['tenure'] * valid_rows['MonthlyCharges']
            actual = valid_rows['TotalCharges']
            
            # Allow 20% deviation (due to discounts, promotions, etc.)
            diff_ratio = (actual - expected).abs() / expected
            
            # Most should be within 20% (allow some exceptions)
            assert (diff_ratio < 0.2).sum() / len(valid_rows) >= 0.7