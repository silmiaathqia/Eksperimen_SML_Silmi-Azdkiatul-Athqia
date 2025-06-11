"""
Remote Worker Productivity Dataset Preprocessing Automation
Author: Silmi Azdkiatul Athqia
Description: Automated preprocessing pipeline for remote worker productivity dataset
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class RemoteWorkerDataPreprocessor:
    """
    Automated preprocessing pipeline for Remote Worker Productivity Dataset
    
    This class handles the complete preprocessing workflow including:
    - Data loading and validation
    - Missing values and duplicates handling
    - Feature encoding
    - Data splitting
    - Feature scaling
    - Model artifacts saving
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the preprocessor
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.label_encoder = None
        self.feature_scaler = None
        self.feature_names = None
        self.preprocessing_info = {}
        
        # Set random seeds
        np.random.seed(self.random_state)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If file cannot be read
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)
            
            print(f"Dataset loaded successfully!")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the dataset
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Validated dataset
        """
        print("Validating dataset...")
        
        # Store original shape
        original_shape = df.shape
        
        # Check for missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"Found {missing_before} missing values - removing rows with missing data")
            df = df.dropna()
        
        # Check for duplicates
        duplicates_before = df.duplicated().sum()
        if duplicates_before > 0:
            print(f"Found {duplicates_before} duplicate rows - removing duplicates")
            df = df.drop_duplicates()
        
        # Store validation info
        self.preprocessing_info['original_shape'] = original_shape
        self.preprocessing_info['missing_values_removed'] = missing_before
        self.preprocessing_info['duplicates_removed'] = duplicates_before
        self.preprocessing_info['final_shape_after_validation'] = df.shape
        
        print(f"Validation complete. Shape: {original_shape} -> {df.shape}")
        
        return df
    
    def remove_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns that are not needed for modeling
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with unnecessary columns removed
        """
        columns_to_drop = ['worker_id', 'productivity_score']
        
        print(f"Removing unnecessary columns: {columns_to_drop}")
        
        # Only drop columns that exist
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df_clean = df.drop(columns=existing_columns_to_drop, errors='ignore')
        
        self.preprocessing_info['columns_dropped'] = existing_columns_to_drop
        
        print(f"Columns removed: {existing_columns_to_drop}")
        print(f"Remaining columns: {list(df_clean.columns)}")
        
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        categorical_features = ['location_type', 'industry_sector']
        
        print(f"Encoding categorical features: {categorical_features}")
        
        # Verify categorical features exist
        existing_categorical = [col for col in categorical_features if col in df.columns]
        
        if not existing_categorical:
            print("No categorical features found to encode")
            return df
        
        # Perform one-hot encoding
        df_encoded = pd.get_dummies(df, columns=existing_categorical, prefix=existing_categorical)
        
        # Store encoding info
        self.preprocessing_info['categorical_features_encoded'] = existing_categorical
        self.preprocessing_info['shape_after_encoding'] = df_encoded.shape
        
        new_columns = [col for col in df_encoded.columns if col not in df.columns]
        print(f"Created {len(new_columns)} new binary features")
        
        return df_encoded
    
    def encode_target_label(self, df: pd.DataFrame, target_column: str = 'productivity_label') -> pd.DataFrame:
        """
        Encode target labels using LabelEncoder
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of target column
            
        Returns:
            pd.DataFrame: Dataset with encoded target labels
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        print(f"Encoding target label: {target_column}")
        
        # Check original distribution
        original_distribution = df[target_column].value_counts()
        print("Original label distribution:")
        for label, count in original_distribution.items():
            percentage = (count / len(df)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Initialize and fit label encoder
        self.label_encoder = LabelEncoder()
        df['productivity_label_encoded'] = self.label_encoder.fit_transform(df[target_column])
        
        # Create label mapping
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_)))
        
        print("Label mapping:")
        for original, encoded in label_mapping.items():
            print(f"  {original} -> {encoded}")
        
        # Store encoding info
        self.preprocessing_info['label_mapping'] = label_mapping
        self.preprocessing_info['label_classes'] = list(self.label_encoder.classes_)
        self.preprocessing_info['num_classes'] = len(self.label_encoder.classes_)
        
        # Remove original target column
        df_final = df.drop(target_column, axis=1)
        
        return df_final
    
    def split_data(self, df: pd.DataFrame, target_column: str = 'productivity_label_encoded',
                   test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, ...]:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of target column
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set from train+val
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"Splitting data - Test: {test_size*100}%, Val: {val_size*100}% of remaining")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        self.feature_names = list(X.columns)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state, stratify=y_temp
        )
        
        # Store split info
        total_samples = len(X)
        self.preprocessing_info['split_info'] = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_percentage': (len(X_train) / total_samples) * 100,
            'val_percentage': (len(X_val) / total_samples) * 100,
            'test_percentage': (len(X_test) / total_samples) * 100
        }
        
        print(f"Data split completed:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/total_samples*100:.1f}%)")
        print(f"  Val:   {len(X_val)} samples ({len(X_val)/total_samples*100:.1f}%)")
        print(f"  Test:  {len(X_test)} samples ({len(X_test)/total_samples*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Scale features using StandardScaler
        
        Args:
            X_train (pd.DataFrame): Training features
            X_val (pd.DataFrame): Validation features
            X_test (pd.DataFrame): Test features
            
        Returns:
            Tuple: (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        print("Scaling features using StandardScaler...")
        
        # Initialize and fit scaler on training data
        self.feature_scaler = StandardScaler()
        
        # Fit and transform
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print("Feature scaling completed")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_preprocessing_artifacts(self, output_dir: str = "preprocessed_data") -> None:
        """
        Save all preprocessing artifacts and processed data
        
        Args:
            output_dir (str): Directory to save artifacts
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving preprocessing artifacts to '{output_dir}'...")
        
        # Save encoders and scalers
        if self.label_encoder:
            joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
            print("  Saved: label_encoder.pkl")
        
        if self.feature_scaler:
            joblib.dump(self.feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
            print("  Saved: feature_scaler.pkl")
        
        # Save feature names
        if self.feature_names:
            feature_info = pd.DataFrame({
                'feature_name': self.feature_names,
                'feature_index': range(len(self.feature_names))
            })
            feature_info.to_csv(os.path.join(output_dir, 'feature_names.csv'), index=False)
            print("  Saved: feature_names.csv")
        
        # Save label mapping
        if self.label_encoder:
            label_mapping_df = pd.DataFrame({
                'original_label': self.label_encoder.classes_,
                'encoded_label': range(len(self.label_encoder.classes_))
            })
            label_mapping_df.to_csv(os.path.join(output_dir, 'label_mapping.csv'), index=False)
            print("  Saved: label_mapping.csv")
        
        # Save preprocessing summary
        self.preprocessing_info['num_features'] = len(self.feature_names) if self.feature_names else 0
        self.preprocessing_info['scaling_method'] = 'StandardScaler'
        self.preprocessing_info['random_state'] = self.random_state
        
        with open(os.path.join(output_dir, 'preprocessing_summary.json'), 'w') as f:
            json.dump(self.preprocessing_info, f, indent=2, default=str)
        print("  Saved: preprocessing_summary.json")
    
    def save_processed_data(self, X_train_scaled: pd.DataFrame, X_val_scaled: pd.DataFrame, 
                           X_test_scaled: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, 
                           y_test: pd.Series, output_dir: str = "preprocessed_data") -> None:
        """
        Save processed datasets
        
        Args:
            X_train_scaled, X_val_scaled, X_test_scaled: Scaled feature sets
            y_train, y_val, y_test: Target sets
            output_dir (str): Directory to save data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving processed datasets to '{output_dir}'...")
        
        # Combine features and targets
        train_data = pd.concat([X_train_scaled, y_train], axis=1)
        val_data = pd.concat([X_val_scaled, y_val], axis=1)
        test_data = pd.concat([X_test_scaled, y_test], axis=1)
        
        # Save datasets
        train_data.to_csv(os.path.join(output_dir, 'data_train.csv'), index=False)
        val_data.to_csv(os.path.join(output_dir, 'data_validation.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, 'data_test.csv'), index=False)
        
        print(f"  Saved: data_train.csv ({train_data.shape})")
        print(f"  Saved: data_validation.csv ({val_data.shape})")
        print(f"  Saved: data_test.csv ({test_data.shape})")
    
    def preprocess(self, file_path: str, output_dir: str = "preprocessed_data") -> Dict[str, Any]:
        """
        Complete preprocessing pipeline
        
        Args:
            file_path (str): Path to input CSV file
            output_dir (str): Directory to save processed data
            
        Returns:
            Dict: Dictionary containing processed datasets and info
        """
        print("="*60)
        print("REMOTE WORKER PRODUCTIVITY DATASET PREPROCESSING")
        print("="*60)
        
        try:
            # Step 1: Load data
            df = self.load_data(file_path)
            
            # Step 2: Validate and clean data
            df = self.validate_data(df)
            
            # Step 3: Remove unnecessary columns
            df = self.remove_unnecessary_columns(df)
            
            # Step 4: Encode categorical features
            df = self.encode_categorical_features(df)
            
            # Step 5: Encode target labels
            df = self.encode_target_label(df)
            
            # Step 6: Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
            
            # Step 7: Scale features
            X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
            
            # Step 8: Save artifacts and data
            self.save_preprocessing_artifacts(output_dir)
            self.save_processed_data(X_train_scaled, X_val_scaled, X_test_scaled, 
                                   y_train, y_val, y_test, output_dir)
            
            print("="*60)
            print("PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Summary:")
            print(f"  Input file: {file_path}")
            print(f"  Output directory: {output_dir}")
            print(f"  Training samples: {len(X_train_scaled)}")
            print(f"  Validation samples: {len(X_val_scaled)}")
            print(f"  Test samples: {len(X_test_scaled)}")
            print(f"  Number of features: {X_train_scaled.shape[1]}")
            print(f"  Number of classes: {len(self.label_encoder.classes_)}")
            
            return {
                'X_train': X_train_scaled,
                'X_val': X_val_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'preprocessing_info': self.preprocessing_info,
                'label_encoder': self.label_encoder,
                'feature_scaler': self.feature_scaler
            }
            
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            raise


def load_preprocessed_data(data_dir: str = "preprocessed_data") -> Dict[str, Any]:
    """
    Load previously preprocessed data and artifacts
    
    Args:
        data_dir (str): Directory containing preprocessed data
        
    Returns:
        Dict: Dictionary containing loaded datasets and artifacts
    """
    print(f"Loading preprocessed data from '{data_dir}'...")
    
    try:
        # Load datasets
        train_data = pd.read_csv(os.path.join(data_dir, 'data_train.csv'))
        val_data = pd.read_csv(os.path.join(data_dir, 'data_validation.csv'))
        test_data = pd.read_csv(os.path.join(data_dir, 'data_test.csv'))
        
        # Separate features and targets
        X_train = train_data.drop('productivity_label_encoded', axis=1)
        y_train = train_data['productivity_label_encoded']
        X_val = val_data.drop('productivity_label_encoded', axis=1)
        y_val = val_data['productivity_label_encoded']
        X_test = test_data.drop('productivity_label_encoded', axis=1)
        y_test = test_data['productivity_label_encoded']
        
        # Load artifacts
        label_encoder = joblib.load(os.path.join(data_dir, 'label_encoder.pkl'))
        feature_scaler = joblib.load(os.path.join(data_dir, 'feature_scaler.pkl'))
        
        # Load preprocessing info
        with open(os.path.join(data_dir, 'preprocessing_summary.json'), 'r') as f:
            preprocessing_info = json.load(f)
        
        print("Data loaded successfully!")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'feature_scaler': feature_scaler,
            'preprocessing_info': preprocessing_info
        }
        
    except Exception as e:
        raise Exception(f"Error loading preprocessed data: {str(e)}")


# WRAPPER FUNCTION FOR GITHUB ACTIONS COMPATIBILITY
def preprocess_remote_worker_data(file_path: str, save_path: str = None) -> pd.DataFrame:
    """
    Wrapper function for GitHub Actions workflow compatibility
    
    Args:
        file_path (str): Path to raw CSV file
        save_path (str): Path to save the final processed dataset (optional)
        
    Returns:
        pd.DataFrame: Final processed dataset ready for training
    """
    try:
        print("🚀 Starting preprocessing workflow...")
        
        # Initialize preprocessor
        preprocessor = RemoteWorkerDataPreprocessor(random_state=42)
        
        # Run complete preprocessing pipeline
        output_dir = "preprocessing/processed_data"
        result = preprocessor.preprocess(file_path, output_dir)
        
        if result and 'X_train' in result:
            # Combine train, val, test into one final dataset if save_path is provided
            if save_path:
                # Create final combined dataset
                X_train_scaled = result['X_train']
                y_train = result['y_train']
                
                # Combine features and target
                final_dataset = pd.concat([X_train_scaled, y_train], axis=1)
                
                # Save the final dataset
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                final_dataset.to_csv(save_path, index=False)
                print(f"✅ Final dataset saved to: {save_path}")
                
                return final_dataset
            else:
                # Return training set as DataFrame
                return pd.concat([result['X_train'], result['y_train']], axis=1)
        else:
            print("❌ Preprocessing failed!")
            return None
            
    except Exception as e:
        print(f"❌ Error in preprocessing workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function to demonstrate the preprocessing pipeline
    """
    # Example usage
    try:
        # Initialize preprocessor
        preprocessor = RemoteWorkerDataPreprocessor(random_state=42)
        
        # Run preprocessing
        input_file = 'remote_worker_productivity_raw.csv'  # Update path as needed
        output_directory = 'preprocessed_data'
        
        result = preprocessor.preprocess(input_file, output_directory)
        
        print("\nPreprocessing pipeline completed successfully!")
        print("Ready for model training!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure the input file exists and is accessible.")


# Alternative main execution for GitHub Actions
if __name__ == "__main__":
    import sys
    
    # Check if running from GitHub Actions
    if len(sys.argv) > 1 and sys.argv[1] == "--github-actions":
        # GitHub Actions mode
        try:
            result = preprocess_remote_worker_data(
                file_path='remote_worker_productivity_raw.csv',
                save_path='preprocessing/processed_data/data_preprocessed.csv'
            )
            
            if result is not None:
                print(f"✅ Preprocessing completed! Final dataset shape: {result.shape}")
                sys.exit(0)
            else:
                print("❌ Preprocessing failed!")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Fatal error: {str(e)}")
            sys.exit(1)
    else:
        # Normal mode
        main()
