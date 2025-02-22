import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging
from dataclasses import dataclass

# Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters"""
    input_dir: Path = Path("output_files")
    output_dir: Path = Path("analysis_results")
    file_pattern: str = "*.csv"
    
class PharmacyAnalyzer:
    """Analyzer for pharmacy data across Japanese prefectures"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_directories()
        
    def _setup_directories(self):
        """Create output directories if they don't exist"""
        for subdir in ['reports', 'visualizations', 'summaries']:
            (self.config.output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
    def get_prefecture_files(self) -> List[Path]:
        """Get list of available prefecture CSV files"""
        files = list(self.config.input_dir.glob(self.config.file_pattern))
        logger.info(f"Found {len(files)} prefecture files")
        return files
    
    def analyze_prefecture(self, file_path: Path) -> Dict[str, Any]:
        """Analyze single prefecture data"""
        try:
            df = pd.read_csv(file_path)
            prefecture_name = file_path.stem.split('_')[-1]
            logger.info(f"Analyzing {prefecture_name}")
            
            analysis = {
                'prefecture': prefecture_name,
                'basic_stats': self._get_basic_stats(df),
                'quality_metrics': self._check_data_quality(df),
                'service_analysis': self._analyze_services(df),
                'geographical': self._analyze_geographical(df)
            }
            
            self._save_prefecture_report(analysis, prefecture_name)
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path.name}: {e}")
            raise
            
    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics"""
        return {
            'total_pharmacies': len(df),
            'unique_cities': df['city'].nunique() if 'city' in df.columns else 0,
            'columns_present': list(df.columns)
        }
        
    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality"""
        return {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'completeness_ratio': 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        }
        
    def _analyze_services(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pharmacy services"""
        service_columns = [col for col in df.columns if any(
            keyword in col.lower() 
            for keyword in ['service', 'emergency', '24hour', 'night']
        )]
        
        service_stats = {}
        for col in service_columns:
            if df[col].dtype in ['int64', 'float64']:
                service_stats[col] = {
                    'mean': df[col].mean(),
                    'count': df[col].sum()
                }
        return service_stats
        
    def _analyze_geographical(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze geographical distribution"""
        if 'city' not in df.columns:
            return {}
            
        city_counts = df['city'].value_counts()
        return {
            'top_cities': city_counts.head(5).to_dict(),
            'city_distribution': {
                'mean': city_counts.mean(),
                'std': city_counts.std()
            }
        }
        
    def _save_prefecture_report(self, analysis: Dict, prefecture: str):
        """Save individual prefecture report"""
        report_path = (
            self.config.output_dir / 'reports' / 
            f'report_{prefecture}_{self.timestamp}.txt'
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Analysis Report for {prefecture}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for section, data in analysis.items():
                f.write(f"\n{section.upper()}\n")
                f.write("-" * len(section) + "\n")
                f.write(f"{data}\n")
                
    def create_visualizations(self, all_results: List[Dict]):
        """Create comparative visualizations"""
        summary_df = pd.DataFrame([
            {
                'prefecture': r['prefecture'],
                'total_pharmacies': r['basic_stats']['total_pharmacies'],
                'unique_cities': r['basic_stats']['unique_cities'],
                'completeness': r['quality_metrics']['completeness_ratio']
            }
            for r in all_results
        ])
        
        # Pharmacy distribution plot
        plt.figure(figsize=(15, 8))
        sns.barplot(data=summary_df, x='prefecture', y='total_pharmacies')
        plt.title('Number of Pharmacies by Prefecture')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            self.config.output_dir / 'visualizations' / 
            f'pharmacy_distribution_{self.timestamp}.png'
        )
        plt.close()
        
        # Data quality plot
        plt.figure(figsize=(15, 8))
        sns.scatterplot(
            data=summary_df, 
            x='total_pharmacies', 
            y='completeness',
            s=100
        )
        plt.title('Data Completeness vs Pharmacy Count')
        for _, row in summary_df.iterrows():
            plt.annotate(
                row['prefecture'], 
                (row['total_pharmacies'], row['completeness'])
            )
        plt.tight_layout()
        plt.savefig(
            self.config.output_dir / 'visualizations' / 
            f'data_quality_{self.timestamp}.png'
        )
        plt.close()
        
        # Save summary data
        summary_df.to_csv(
            self.config.output_dir / 'summaries' / 
            f'analysis_summary_{self.timestamp}.csv', 
            index=False
        )
        
def main():
    """Main execution function"""
    try:
        # Initialize configuration
        config = AnalysisConfig()
        analyzer = PharmacyAnalyzer(config)
        
        # Get and analyze all prefecture files
        prefecture_files = analyzer.get_prefecture_files()
        if not prefecture_files:
            logger.error("No prefecture files found!")
            return
            
        # Analyze all prefectures
        analysis_results = []
        for file_path in prefecture_files:
            try:
                result = analyzer.analyze_prefecture(file_path)
                analysis_results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {file_path.name}: {e}")
                continue
                
        # Create visualizations and summary
        analyzer.create_visualizations(analysis_results)
        logger.info(
            f"Analysis complete! Results saved in {config.output_dir}"
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()