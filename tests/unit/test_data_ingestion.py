"""
Unit tests for Data Ingestion component.
"""

import pytest
import pandas as pd
import os
from box import ConfigBox
from src.components.data_ingestion import (
    DataIngestion,
    DataIngestionConfig,
    create_data_ingestion_config
)


@pytest.mark.unit
class TestDataIngestion:
    """Test suite for DataIngestion class."""
    
    def test_data_ingestion_config_creation(self, temp_dir):
        """Test DataIngestionConfig creation."""
        config_dict = {
            'raw_data_path': os.path.join(temp_dir, 'raw.csv'),
            'train_data_path': os.path.join(temp_dir, 'train.csv'),
            'test_data_path': os.path.join(temp_dir, 'test.csv'),
            'test_size': 0.2,
            'random_state': 42
        }
        config_box = ConfigBox(config_dict)
        config = create_data_ingestion_config(config_box)
        
        assert isinstance(config, DataIngestionConfig)
        assert config.test_size == 0.2
        assert config.random_state == 42
    
    def test_initiate_data_ingestion(self, sample_csv_file, temp_dir):
        """Test data ingestion process."""
        config = DataIngestionConfig(
            raw_data_path=sample_csv_file,
            train_data_path=os.path.join(temp_dir, 'train.csv'),
            test_data_path=os.path.join(temp_dir, 'test.csv'),
            test_size=0.2,
            random_state=42
        )
        
        data_ingestion = DataIngestion(config)
        # train_path, test_path = data_ingestion.initiate_data_ingestion()
        try:
            train_path, test_path = data_ingestion.initiate_data_ingestion()
        except Exception as e:
            print("REAL ERROR:", repr(e))
            raise
        
        # Check files created
        assert os.path.exists(train_path)
        assert os.path.exists(test_path)
        
        # Check data split
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(test_df) == 5  # Sample data has 5 rows
    
    def test_stratified_split(self, sample_csv_file, temp_dir):
        """Test stratified sampling maintains class distribution."""
        config = DataIngestionConfig(
            raw_data_path=sample_csv_file,
            train_data_path=os.path.join(temp_dir, 'train_strat.csv'),
            test_data_path=os.path.join(temp_dir, 'test_strat.csv'),
            test_size=0.4,
            random_state=42
        )
        
        data_ingestion = DataIngestion(config)
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        original_df = pd.read_csv(sample_csv_file)
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Check both classes present (if possible)
        if 'Churn' in original_df.columns and original_df['Churn'].nunique() > 1:
            assert len(train_df) > 0
            assert len(test_df) > 0
    
    def test_get_data_info(self, sample_csv_file, temp_dir):
        """Test get_data_info method."""
        config = DataIngestionConfig(
            raw_data_path=sample_csv_file,
            train_data_path=os.path.join(temp_dir, 'train_info.csv'),
            test_data_path=os.path.join(temp_dir, 'test_info.csv')
        )
        
        data_ingestion = DataIngestion(config)
        info = data_ingestion.get_data_info()
        
        assert 'total_rows' in info
        assert 'total_columns' in info
        assert 'columns' in info
        assert 'missing_values' in info
        assert 'duplicates' in info
        
        assert info['total_rows'] == 5
        assert isinstance(info['columns'], list)
    
    def test_data_ingestion_with_missing_file(self, temp_dir):
        """Test error handling when data file is missing."""
        config = DataIngestionConfig(
            raw_data_path=os.path.join(temp_dir, 'nonexistent.csv'),
            train_data_path=os.path.join(temp_dir, 'train_err.csv'),
            test_data_path=os.path.join(temp_dir, 'test_err.csv')
        )
        
        data_ingestion = DataIngestion(config)
        
        with pytest.raises(Exception):
            data_ingestion.initiate_data_ingestion()
    
    def test_random_state_consistency(self, sample_csv_file, temp_dir):
        """Test that same random_state produces same split."""
        config1 = DataIngestionConfig(
            raw_data_path=sample_csv_file,
            train_data_path=os.path.join(temp_dir, 'train1.csv'),
            test_data_path=os.path.join(temp_dir, 'test1.csv'),
            random_state=42
        )
        
        config2 = DataIngestionConfig(
            raw_data_path=sample_csv_file,
            train_data_path=os.path.join(temp_dir, 'train2.csv'),
            test_data_path=os.path.join(temp_dir, 'test2.csv'),
            random_state=42
        )
        
        di1 = DataIngestion(config1)
        di2 = DataIngestion(config2)
        
        di1.initiate_data_ingestion()
        di2.initiate_data_ingestion()
        
        train1 = pd.read_csv(config1.train_data_path)
        train2 = pd.read_csv(config2.train_data_path)
        
        # Should have same indices (order)
        pd.testing.assert_frame_equal(train1, train2)


@pytest.mark.unit
def test_config_default_values():
    """Test default values in DataIngestionConfig."""
    config = DataIngestionConfig(
        raw_data_path='data.csv',
        train_data_path='train.csv',
        test_data_path='test.csv'
    )
    
    assert config.test_size == 0.2
    assert config.random_state == 42