import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List, Dict
import time
import logging

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GradientDescent:
    """
    Implementación del algoritmo de descenso del gradiente para regresión lineal.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01, 
                 epochs: int = 1000, 
                 tolerance: float = 1e-6,
                 verbose: bool = True) -> None:
        """
        Inicializa el modelo de descenso del gradiente.
        
        Args:
            learning_rate: Tasa de aprendizaje para actualizar parámetros
            epochs: Número máximo de iteraciones
            tolerance: Criterio de convergencia
            verbose: Si True, muestra información durante el entrenamiento
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.verbose = verbose
        self.costs = []
        self.w = None
        self.b = None
        self.training_history = {
            'epoch': [],
            'cost': [],
            'w': [],
            'b': [],
            'time': []
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando el modelo actual.
        
        Args:
            X: Array de características de entrada
            
        Returns:
            Array con predicciones
        """
        if self.w is None or self.b is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        return np.dot(X, self.w) + self.b
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula el costo (error cuadrático medio) del modelo actual.
        
        Args:
            X: Array de características
            y: Array de valores objetivo
            
        Returns:
            Valor del costo
        """
        m = len(X)
        predictions = self.predict(X)
        cost = np.sum((predictions - y) ** 2) / (2 * m)
        return cost
    
    def compute_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas adicionales de rendimiento.
        
        Args:
            X: Array de características
            y: Array de valores objetivo
            
        Returns:
            Diccionario con métricas calculadas
        """
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        r2 = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientDescent':
        """
        Entrena el modelo usando descenso del gradiente.
        
        Args:
            X: Array de características de entrenamiento
            y: Array de valores objetivo
            
        Returns:
            Self para encadenamiento de métodos
        """
        # Inicio del tiempo de entrenamiento
        start_time = time.time()
        
        # Inicialización de parámetros
        m = len(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        logging.info(f"Iniciando entrenamiento con {n_features} características y {m} ejemplos")
        
        # Ciclo principal de entrenamiento
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Forward pass
            predictions = self.predict(X)
            
            # Cálculo de gradientes
            dw = np.dot(X.T, (predictions - y)) / m
            db = np.sum(predictions - y) / m
            
            # Actualización de parámetros
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Cálculo y almacenamiento del costo
            cost = self.compute_cost(X, y)
            self.costs.append(cost)
            
            # Almacenamiento del historial de entrenamiento
            self.training_history['epoch'].append(epoch)
            self.training_history['cost'].append(cost)
            self.training_history['w'].append(self.w.copy())
            self.training_history['b'].append(self.b)
            self.training_history['time'].append(time.time() - epoch_start_time)
            
            # Logging del progreso
            if self.verbose and epoch % 100 == 0:
                metrics = self.compute_metrics(X, y)
                logging.info(
                    f"Época {epoch}/{self.epochs} - "
                    f"Costo: {cost:.6f} - "
                    f"MSE: {metrics['MSE']:.6f} - "
                    f"R2: {metrics['R2']:.6f}"
                )
            
            # Verificación de convergencia
            if epoch > 0 and abs(self.costs[-1] - self.costs[-2]) < self.tolerance:
                logging.info(f"Convergencia alcanzada en época {epoch}")
                break
        
        training_time = time.time() - start_time
        logging.info(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        return self
    
    def plot_convergence(self) -> None:
        """
        Visualiza la curva de convergencia del entrenamiento.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.costs)
        plt.title('Curva de Convergencia del Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Costo')
        plt.grid(True)
        plt.yscale('log')
        plt.show()
    
    def plot_regression_line(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Visualiza la línea de regresión ajustada.
        
        Args:
            X: Array de características
            y: Array de valores objetivo
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', alpha=0.5, label='Datos')
        plt.plot(X, self.predict(X), color='red', label='Regresión')
        plt.title('Línea de Regresión Ajustada')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_residuals(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Visualiza el gráfico de residuos.
        
        Args:
            X: Array de características
            y: Array de valores objetivo
        """
        predictions = self.predict(X)
        residuals = y - predictions
        
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Gráfico de Residuos')
        plt.xlabel('Predicciones')
        plt.ylabel('Residuos')
        plt.grid(True)
        plt.show()
    
    def save_model(self, filename: str) -> None:
        """
        Guarda los parámetros del modelo en un archivo.
        
        Args:
            filename: Nombre del archivo para guardar
        """
        model_params = {
            'w': self.w,
            'b': self.b,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'tolerance': self.tolerance,
            'costs': self.costs
        }
        np.save(filename, model_params)
        logging.info(f"Modelo guardado en {filename}")
    
    def load_model(self, filename: str) -> None:
        """
        Carga los parámetros del modelo desde un archivo.
        
        Args:
            filename: Nombre del archivo para cargar
        """
        model_params = np.load(filename, allow_pickle=True).item()
        self.w = model_params['w']
        self.b = model_params['b']
        self.learning_rate = model_params['learning_rate']
        self.epochs = model_params['epochs']
        self.tolerance = model_params['tolerance']
        self.costs = model_params['costs']
        logging.info(f"Modelo cargado desde {filename}")

def generate_sample_data(n_samples: int = 100, 
                        noise: float = 0.1, 
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos de ejemplo para regresión lineal.
    
    Args:
        n_samples: Número de muestras a generar
        noise: Nivel de ruido en los datos
        seed: Semilla para reproducibilidad
        
    Returns:
        Tupla con arrays (X, y)
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, 1)
    y = 2 * X + 1 + np.random.randn(n_samples, 1) * noise
    return X.reshape(-1), y.reshape(-1)

def main():
    """
    Función principal para demostrar el uso del algoritmo.
    """
    # Generar datos de ejemplo
    X, y = generate_sample_data(n_samples=100, noise=0.1)
    
    # Crear y entrenar modelo
    model = GradientDescent(
        learning_rate=0.01,
        epochs=1000,
        tolerance=1e-6,
        verbose=True
    )
    
    # Entrenamiento
    model.fit(X, y)
    
    # Visualizaciones
    model.plot_convergence()
    model.plot_regression_line(X, y)
    model.plot_residuals(X, y)
    
    # Métricas finales
    final_metrics = model.compute_metrics(X, y)
    logging.info("Métricas finales:")
    for metric_name, metric_value in final_metrics.items():
        logging.info(f"{metric_name}: {metric_value:.6f}")
    
    # Guardar modelo
    model.save_model('gradient_descent_model.npy')

if __name__ == "__main__":
    main()