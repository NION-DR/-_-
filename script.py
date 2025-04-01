import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple

class BoxplotGenerator:
    """Генератор боксплотов с автоматическим определением масштаба"""
    
    def __init__(self, data: pd.DataFrame, column: str, output_dir: str, 
                 limits: Optional[Tuple[float, float]] = None):
        self.data = data
        self.column = column
        self.output_dir = output_dir
        self.limits = limits

    def _determine_scale(self, series: pd.Series, stats: dict) -> str:
        """Автоматический выбор шкалы для визуализации"""
        data_range = stats['upper'] - stats['lower']
        if (series.max() - stats['upper']) > 10*data_range or (stats['lower'] - series.min()) > 10*data_range:
            return 'log' if stats['lower'] > 0 else 'symlog'
        return 'linear'

    def _calculate_statistics(self, series: pd.Series) -> dict:
        """Расчет статистик для боксплота"""
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
    """Генератор гистограмм с выделением выбросов"""
    
    def __init__(self, data: pd.DataFrame, column: str, output_dir: str,
                 limits: Optional[Tuple[float, float]] = None):
        self.data = data
        self.column = column
        self.output_dir = output_dir
        self.limits = limits

    def _calculate_statistics(self, series: pd.Series) -> dict:
        """Расчет статистик для гистограммы"""
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
    """Генератор точечных графиков с субсэмплингом"""
    
    def __init__(self, data: pd.DataFrame, x_col: str, y_col: str, 
                 output_dir: str, x_limits: Optional[Tuple[float, float]] = None,
                 y_limits: Optional[Tuple[float, float]] = None):
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.output_dir = output_dir
        self.x_limits = x_limits
        self.y_limits = y_limits

    def _subsample_data(self, df: pd.DataFrame, x_min: float, x_max: float,
                       y_min: float, y_max: float) -> pd.DataFrame:
        """Субсэмплинг данных для визуализации"""
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

    def generate(self):
        """Генерация и сохранение точечного графика"""
        clean_df = self.data[[self.x_col, self.y_col]].dropna()
        if clean_df.empty:
            raise ValueError("Нет данных для построения")

        x_min, x_max = self.x_limits or (clean_df[self.x_col].min(), clean_df[self.x_col].max())
        y_min, y_max = self.y_limits or (clean_df[self.y_col].min(), clean_df[self.y_col].max())

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

class Visualizer:
    """Основной интерфейс для работы с графиками"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Автоматическое определение формата файла"""
        if self.file_path.endswith('.tsv'):
            return pd.read_csv(self.file_path, sep='\t')
        elif self.file_path.endswith('.csv'):
            return pd.read_csv(self.file_path)
        raise ValueError("Поддерживаются только CSV/TSV файлы")

    def create_plot(self, plot_type: str, **kwargs):
        """
        Создание графика со следующими параметрами:
        - plot_type: 'hist', 'box', 'scatter'
        - column: для hist/box
        - x_col/y_col: для scatter
        - limits: для hist/box
        - x_limits/y_limits: для scatter
        - output_dir: папка для сохранения
        """
        try:
            output_dir = kwargs.get('output_dir', 'visualizations')
            os.makedirs(output_dir, exist_ok=True)

            if plot_type in ['hist', 'box']:
                self._create_single_plot(plot_type, output_dir, kwargs)
            elif plot_type == 'scatter':
                self._create_scatter_plot(output_dir, kwargs)
            else:
                raise ValueError("Неподдерживаемый тип графика")

            print(f"✓ График успешно сохранен в: {output_dir}")
        
        except Exception as e:
            print(f"✗ Ошибка: {str(e)}")
            exit(1)

    def _create_single_plot(self, plot_type: str, output_dir: str, params: dict):
        column = params.get('column')
        if not column:
            raise ValueError("Необходимо указать column")

        limits = params.get('limits')
        
        if plot_type == 'hist':
            HistogramGenerator(self.data, column, output_dir, limits).generate()
        else:
            BoxplotGenerator(self.data, column, output_dir, limits).generate()

    def _create_scatter_plot(self, output_dir: str, params: dict):
        x_col = params.get('x_col')
        y_col = params.get('y_col')
        if not x_col or not y_col:
            raise ValueError("Необходимо указать x_col и y_col")

        ScatterPlotGenerator(
            self.data,
            x_col,
            y_col,
            output_dir,
            params.get('x_limits'),
            params.get('y_limits')
        ).generate()

def get_file_path():
    """Получение пути к файлу данных"""
    while True:
        path = input("▸ Введите путь к файлу данных (*.csv/*.tsv): ").strip()
        if not os.path.isfile(path):
            print("✗ Файл не найден!")
            continue
        if not (path.endswith('.csv') or path.endswith('.tsv')):
            print("✗ Поддерживаются только CSV/TSV файлы!")
            continue
        return path

def get_plot_type():
    """Выбор типа графика"""
    while True:
        plot_type = input("▸ Выберите тип графика (hist/box/scatter): ").lower().strip()
        if plot_type in ['hist', 'box', 'scatter']:
            return plot_type
        print("✗ Некорректный тип! Допустимые значения: hist, box, scatter")

def get_column(columns: list, purpose: str) -> str:
    """Выбор столбца из списка"""
    while True:
        print("\nДоступные столбцы:", ', '.join(columns))
        col = input(f"▸ Введите название столбца для {purpose}: ").strip()
        if col in columns:
            return col
        print("✗ Столбец не найден!")

def get_limits(axis_name: str) -> Optional[Tuple[float, float]]:
    """Ввод границ значений"""
    while True:
        try:
            limits = input(
                f"▸ Введите границы для {axis_name} (мин макс) или Enter для автоопределения: "
            ).strip()
            if not limits:
                return None
            lower, upper = map(float, limits.split())
            if lower >= upper:
                print("✗ Нижняя граница должна быть меньше верхней!")
                continue
            return (lower, upper)
        except:
            print("✗ Некорректный ввод! Пример: 0 100 или 50.5 200.75")

def main():
    """Выполнение программы"""
    # Шаг 1: Получение файла данных
    file_path = get_file_path()
    visualizer = Visualizer(file_path)
    
    # Шаг 2: Выбор типа графика
    plot_type = get_plot_type()
    
    # Шаг 3: Сбор параметров
    params = {}
    columns = visualizer.data.columns.tolist()
    
    if plot_type in ['hist', 'box']:
        params['column'] = get_column(columns, "анализа")
        params['limits'] = get_limits("значений")
    else:
        params['x_col'] = get_column(columns, "оси X")
        params['y_col'] = get_column(columns, "оси Y")
        params['x_limits'] = get_limits("оси X")
        params['y_limits'] = get_limits("оси Y")
    
    # Шаг 4: Папка для сохранения
    output_dir = input("\n▸ Введите папку для сохранения (по умолчанию visualizations): ").strip()
    if output_dir:
        params['output_dir'] = output_dir
    
    # Создание графика
    visualizer.create_plot(plot_type, **params)

if __name__ == "__main__":
    main()
