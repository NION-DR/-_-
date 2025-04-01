import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from typing import Optional, Tuple

class BoxplotGenerator:
    """Визуализация распределения данных через квартили и выбросы"""
    
    def __init__(self, data: pd.DataFrame, column: str, output_dir: str, 
                 limits: Optional[Tuple[float, float]] = None):
        self.data = data
        self.column = column
        self.output_dir = output_dir
        self.limits = limits
        os.makedirs(output_dir, exist_ok=True)

    def _determine_scale(self, series: pd.Series, stats: dict) -> str:
        """Выбор масштаба оси Y на основе диапазона данных"""   
        data_range = stats['upper'] - stats['lower']
        if (series.max() - stats['upper']) > 10*data_range or (stats['lower'] - series.min()) > 10*data_range:
            return 'log' if stats['lower'] > 0 else 'symlog'
        return 'linear'

    def _calculate_statistics(self, series: pd.Series) -> dict:
        """Расчет ключевых статистик: квартили, медиана, границы выбросов"""
        stats = {
            'Q1': np.percentile(series, 25),
            'Q3': np.percentile(series, 75),
            'median': series.median(),
            'lower': None,
            'upper': None,
            'outliers': 0
        }
        
        if self.limits:
            stats['lower'], stats['upper'] = self.limits
        else:
            IQR = stats['Q3'] - stats['Q1']
            stats['lower'] = max(stats['Q1'] - 1.5*IQR, series.min())
            stats['upper'] = min(stats['Q3'] + 1.5*IQR, series.max())
        
        stats['outliers'] = ((series < stats['lower']) | (series > stats['upper'])).sum()
        return stats

    def generate(self):
        """Генерация и сохранение боксплота"""
        series = self.data[self.column].dropna()
        if series.empty:
            raise ValueError(f"Столбец '{self.column}' не содержит данных")

        stats = self._calculate_statistics(series)
        
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        box_style = {
            'patch_artist': True,
            'boxprops': dict(facecolor='#8DA0CB', linewidth=2),
            'whiskerprops': dict(linewidth=1.5, color='#666666'),
            'medianprops': dict(color='#E7298A', linewidth=3),
            'flierprops': dict(marker='o', markerfacecolor='#FC8D62', markersize=5, alpha=0.5)
        }
        
        ax.boxplot(series.dropna(), **box_style)
        ax.set_yscale(self._determine_scale(series, stats))
        
        ax.axhline(stats['lower'], color='#66C2A5', linestyle='--', linewidth=2)
        ax.axhline(stats['upper'], color='#66C2A5', linestyle='--', linewidth=2)
        
        stats_text = (f"Медиана: {stats['median']:.2f}\nQ1: {stats['Q1']:.2f}\n"
                     f"Q3: {stats['Q3']:.2f}\nВыбросы: {stats['outliers']}")
        plt.text(1.08, 0.5, stats_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
        
        plt.subplots_adjust(right=0.85)
        ax.set_title(f"Boxplot: {self.column}", fontsize=14)
        plt.savefig(os.path.join(self.output_dir, f"boxplot_{self.column}.svg"), bbox_inches='tight')
        plt.close()

class HistogramGenerator:
    """Анализ распределения данных с выделением аномальных значений"""
    
    def __init__(self, data: pd.DataFrame, column: str, output_dir: str,
                 limits: Optional[Tuple[float, float]] = None):
        self.data = data
        self.column = column
        self.output_dir = output_dir
        self.limits = limits
        os.makedirs(output_dir, exist_ok=True)

    def _calculate_statistics(self, series: pd.Series) -> dict:
        """Определение границ и подсчет выбросов (ручные или автоматические)"""
        stats = {
            'lower': self.limits[0] if self.limits else series.min(),
            'upper': self.limits[1] if self.limits else series.max(),
            'outliers': 0
        }
        
        if self.limits:
            stats['outliers'] = ((series < self.limits[0]) | (series > self.limits[1])).sum()
        return stats

    def generate(self):
        """Генерация и сохранение гистограммы"""
        series = self.data[self.column].dropna()
        if series.empty:
            raise ValueError(f"Столбец '{self.column}' не содержит данных")

        stats = self._calculate_statistics(series)
        
        plt.figure(figsize=(10, 7))
        main_data = series[(series >= stats['lower']) & (series <= stats['upper'])]
        outliers = series[(series < stats['lower']) | (series > stats['upper'])]
        
        bins = np.linspace(stats['lower'], stats['upper'], 50)
        plt.hist(main_data, bins=bins, color='skyblue', edgecolor='black', label='Основные данные')
        plt.hist(outliers, bins=50, color='red', alpha=0.5, label='Выбросы')
        
        plt.axvline(stats['lower'], color='green', linestyle='--', linewidth=2)
        plt.axvline(stats['upper'], color='green', linestyle='--', linewidth=2)
        plt.title(f"Histogram: {self.column}\nВыбросы: {stats['outliers']}", fontsize=14)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"hist_{self.column}.svg"))
        plt.close()

class ScatterPlotGenerator:
    """Визуализация взаимосвязи двух переменных с оптимизацией отображения"""
    
    def __init__(self, data: pd.DataFrame, x_col: str, y_col: str, 
                 output_dir: str, x_limits: Optional[Tuple[float, float]] = None,
                 y_limits: Optional[Tuple[float, float]] = None):
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.output_dir = output_dir
        self.x_limits = x_limits
        self.y_limits = y_limits
        os.makedirs(output_dir, exist_ok=True)

    def generate(self):
        """Генерация и сохранение точечного графика"""
        clean_df = self.data[[self.x_col, self.y_col]].dropna()
        if clean_df.empty:
            raise ValueError("Нет данных для построения после удаления NaN")

        # Определение лимитов
        x_min, x_max = self.x_limits or (clean_df[self.x_col].min(), clean_df[self.x_col].max())
        y_min, y_max = self.y_limits or (clean_df[self.y_col].min(), clean_df[self.y_col].max())

        # Субсэмплинг данных
        subsample = self._subsample_data(clean_df, x_min, x_max, y_min, y_max)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(subsample[self.x_col], subsample[self.y_col], 
                   alpha=0.6, edgecolor='w', linewidth=0.5, s=10)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Scatter: {self.x_col} vs {self.y_col}", fontsize=14)
        plt.xlabel(self.x_col)
        plt.ylabel(self.y_col)
        plt.grid(alpha=0.2)
        
        filename = f"scatter_{self.x_col}_vs_{self.y_col}.svg"
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()

    def _subsample_data(self, df: pd.DataFrame, x_min: float, x_max: float,
                       y_min: float, y_max: float) -> pd.DataFrame:
        """Субсэмплинг данных: сохранение выбросов при сокращении основного набора"""
        target_size = max(1, len(df) // 20)
        inliers = df[
            (df[self.x_col].between(x_min, x_max)) & 
            (df[self.y_col].between(y_min, y_max))
        ]
        outliers = df[
            (~df[self.x_col].between(x_min, x_max)) |
            (~df[self.y_col].between(y_min, y_max))
        ]
        
        if len(inliers) > target_size:
            return pd.concat([inliers.sample(target_size), outliers])
        return pd.concat([inliers, outliers.sample(min(target_size, len(outliers)))])

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Генератор визуализаций для CSV файлов',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Базовые параметры для всех графиков
    parser.add_argument('plot_type', choices=['boxplot', 'histogram', 'scatter'],
                       help='Тип визуализации')
    parser.add_argument('--data', required=True, 
                       help='Путь к CSV файлу с данными')
    parser.add_argument('--sep', default=',', 
                       help='Разделитель в CSV файле')
    parser.add_argument('--output', default='visualizations', 
                       help='Выходная директория для графиков')
    
    # Параметры для графиков с одним столбцом
    parser.add_argument('--column', 
                       help='Имя столбца для анализа (для boxplot и histogram)')
    parser.add_argument('--limits', nargs=2, type=float, 
                       help='Границы значений в формате "мин макс"')
    
    # Параметры для scatter plot
    parser.add_argument('--x-col', 
                       help='Столбец для оси X (только для scatter)')
    parser.add_argument('--y-col', 
                       help='Столбец для оси Y (только для scatter)')
    parser.add_argument('--x-limits', nargs=2, type=float, 
                       help='Границы по оси X в формате "мин макс"')
    parser.add_argument('--y-limits', nargs=2, type=float,
                       help='Границы по оси Y в формате "мин макс"')

    args = parser.parse_args()

    try:
        # Загрузка данных из файла
        df = pd.read_csv(args.data, sep=args.sep)
        print(f"Успешно загружено {len(df)} строк из файла {args.data}")
        
        # Создание визуализации в зависимости от типа графика
        if args.plot_type in ['boxplot', 'histogram']:
            if not args.column:
                raise ValueError("Для этого типа графика необходимо указать --column")
            
            if args.plot_type == 'boxplot':
                generator = BoxplotGenerator(
                    df, args.column, args.output, args.limits
                )
            else:
                generator = HistogramGenerator(
                    df, args.column, args.output, args.limits
                )
        
        elif args.plot_type == 'scatter':
            if not args.x_col or not args.y_col:
                raise ValueError("Для scatter plot необходимо указать --x-col и --y-col")
            
            generator = ScatterPlotGenerator(
                df, args.x_col, args.y_col, args.output,
                args.x_limits, args.y_limits
            )
        
        # Генерация и сохранение графика
        generator.generate()
        print(f"График успешно сохранен в директорию: {args.output}")
    
    except FileNotFoundError:
        print(f"Ошибка: файл {args.data} не найден")
        exit(1)
    except pd.errors.ParserError:
        print(f"Ошибка чтения файла. Проверьте правильность разделителя (--sep)")
        exit(1)
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()

"""
Примеры для создания визуализаций
python script.py histogram --data loco_11_corr.tsv --column "loco_11.tu17l2" --sep "\t"
python script.py histogram --data loco_11_corr.tsv --column "loco_11.tu17l2" --limits 0 2000000 --sep "\t"
python script.py scatter --data loco_11_corr.tsv --x-col "loco_11.tu17l2" --y-col "loco_11.tu17l3" --sep "\t" --x-limits 0 1500 --y-limits 0 2000
python script.py boxplot --data loco_11_corr.tsv --column "loco_11.tu17l3" --limits 500 1500 --sep "\t"
python script.py scatter --data loco_11_corr.tsv --x-col "loco_11.tu17l2" --y-col "loco_11.tu17l3" --sep "\t"
"""
