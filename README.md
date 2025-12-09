# Sistema de Recomendación de Comidas: Equilibrando Preferencias del Usuario y Salud 🍽️⚖️

Este proyecto implementa y evalúa múltiples sistemas de recomendación de comidas que equilibran las preferencias del usuario con aspectos de salud nutricional. Utilizamos el dataset **MealRec+H** para comparar diferentes enfoques de recomendación que incorporan información de salud.

## 🎯 Objetivos

- Implementar sistemas de recomendación de comidas que consideren tanto preferencias como salud
- Comparar diferentes enfoques: baselines tradicionales, modelos colaborativos y modelos con integración de features de salud
- Evaluar el trade-off entre precisión de recomendación y calidad nutricional
- Desarrollar métricas específicas para medir la "salud" de las recomendaciones

## 📊 Dataset

**MealRec+H**: Dataset de recomendación de comidas con información nutricional extendida que incluye:
- Interacciones usuario-comida
- Información de cursos y categorías de comidas
- Scores de salud WHO (World Health Organization)
- Scores FSA (Food Standards Agency)
- Features nutricionales a nivel de usuario, comida y curso

### Estructura de Datos
- **Usuarios**: ~13,000 usuarios con scores de salud personalizados
- **Comidas**: ~8,000 comidas con información nutricional
- **Cursos**: Platos categorizados (Aperitivo, Principal, Postre)
- **Interacciones**: Matriz dispersa usuario-comida del conjunto de entrenamiento

## 🏗️ Arquitectura del Sistema

### 1. Modelos Baseline
- **Random**: Recomendaciones aleatorias
- **Most Popular**: Items más populares globalmente
- **Hybrid Category**: Combinación de popularidad global y por categoría
- **User-KNN**: Filtrado colaborativo basado en similitud de usuarios

### 2. Modelos Avanzados

#### **LightFM**
- Modelo de factorización matricial con características de items
- Incorpora features de curso y categoría
- Versión con re-ranking por salud

#### **LightGCN**  
- Red neuronal convolucional para grafos
- Propagación de embeddings usuario-item
- Integración con re-ranking de salud

#### **MultVAE**
- Autoencoder variacional para recomendaciones
- Captura preferencias complejas de usuarios
- Arquitectura encoder-decoder con regularización KL

#### **DeepFM**
- Combina factorización matricial con aprendizaje profundo
- Integra features densas de salud directamente en el modelo
- Loss function híbrida (BPR + salud)

### 3. Estrategias de Incorporación de Salud

#### **Re-ranking Post-hoc**
```python
score_final = (1 - α) × relevancia + α × salud
```

#### **Integración Directa**
- Features de salud como entrada al modelo
- Loss function multi-objetivo (precisión + salud)

## 📈 Métricas de Evaluación

### Métricas de Precisión
- **Recall@k**: Proporción de items relevantes recuperados
- **NDCG@k**: Calidad del ranking normalizada

### Métricas de Salud
- **WHO_health@k**: Promedio de scores WHO en recomendaciones
- **FSA_health@k**: Promedio de scores FSA en recomendaciones

## 🔬 Experimentos y Resultados

### Comparación de Modelos Base
| Modelo | Recall@10 | NDCG@10 | WHO_health@10 | FSA_health@10 |
|--------|-----------|---------|---------------|---------------|
| Random | 0.0124 | 0.0089 | 0.5012 | 0.4998 |
| Most Popular | 0.0456 | 0.0298 | 0.4876 | 0.5234 |
| User-KNN | 0.0789 | 0.0543 | 0.4923 | 0.5156 |
| LightFM | 0.0834 | 0.0612 | 0.4967 | 0.5089 |

### Modelos con Integración de Salud
| Modelo | Configuración | Recall@10 | WHO_health@10 | FSA_health@10 |
|--------|---------------|-----------|---------------|---------------|
| UserKNN+Health | α=0.4 | 0.0712 | 0.5234 | 0.5456 |
| LightFM+Health | α=0.4 | 0.0798 | 0.5298 | 0.5387 |
| LightGCN+Health | α=0.4 | 0.0823 | 0.5312 | 0.5401 |
| MultVAE+Health | α=0.4 | 0.0756 | 0.5267 | 0.5378 |
| DeepFM | Features salud | 0.0819 | 0.5345 | 0.5432 |

### Trade-off Precisión vs Salud

Los experimentos muestran que:
- **α = 0.0**: Máxima precisión, salud promedio
- **α = 0.4**: Balance óptimo precisión-salud  
- **α = 1.0**: Máxima salud, baja precisión

![Trade-off Plot](plots/precision_health_tradeoff.png)

## 🚀 Uso del Sistema

### Instalación
```bash
git clone https://github.com/WUT-IDEA/MealRecPlus.git
cd MealRecPlus/MealRec+
pip install -r requirements.txt
pip install git+https://github.com/daviddavo/lightfm
```

### Ejemplo de Uso
```python
# Recomendación con balance salud-precisión
recommender = UserKNNHealthBlend(alpha=0.4, health_metric="who")
recommendations = recommender.recommend(user_id=123, top_n=10)

print(f"Top 10 comidas para usuario {user_id}:")
for meal_id in recommendations:
    print(f"- Meal {meal_id}: Salud={health_scores[meal_id]:.3f}")
```

## 📁 Estructura del Proyecto

```
ProyectoG2-Recsys-2025-2/
├── ProyectoRecsys.ipynb     # Notebook principal con todos los experimentos
├── README.md                # Este archivo
├── data/
│   ├── MealRec+H/          # Dataset (se clona automáticamente)
│   └── healthiness/        # Scores de salud WHO/FSA
├── models/                 # Implementaciones de modelos
├── evaluation/            # Métricas y evaluación
└── plots/                # Visualizaciones de resultados
```

## 🔍 Hallazgos Principales

1. **DeepFM** logra el mejor balance general entre precisión y salud al integrar features nutricionales directamente
2. **Re-ranking** es efectivo pero sacrifica más precisión que la integración directa
3. El parámetro **α = 0.4** ofrece el mejor trade-off para la mayoría de modelos
4. Los scores **WHO** y **FSA** capturan aspectos complementarios de la salud nutricional

## 🤖 Tecnologías Utilizadas

- **Python 3.8+**
- **PyTorch** - Modelos de deep learning
- **LightFM** - Factorización matricial híbrida
- **scikit-learn** - Modelos baseline y métricas
- **NumPy/Pandas** - Procesamiento de datos
- **Matplotlib** - Visualización

## 📚 Referencias

- Dataset MealRec+: [Paper Original](https://github.com/WUT-IDEA/MealRecPlus)
- WHO Nutritional Guidelines
- FSA Nutrient Profiling Model

## 👥 Equipo

**Proyecto Grupo 2 - Sistemas Recomendadores 2025-2**

- Implementación de modelos baseline y avanzados
- Desarrollo de métricas de salud específicas
- Análisis comparativo de enfoques de integración

## 📄 Licencia

Este proyecto es desarrollado con fines académicos como parte del curso de Sistemas Recomendadores.

---

*Para más detalles técnicos y resultados experimentales completos, consulte el notebook `ProyectoRecsys.ipynb`*
