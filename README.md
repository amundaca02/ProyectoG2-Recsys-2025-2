# Sistema de Recomendación de Comidas: Equilibrando Preferencias del Usuario y Salud

Este proyecto implementa y evalúa múltiples sistemas de recomendación de comidas que equilibran las preferencias del usuario con aspectos de salud nutricional. Utilizamos el dataset **MealRec+H** para comparar diferentes enfoques de recomendación que incorporan información de salud.

## Objetivos

- Implementar sistemas de recomendación de comidas que consideren tanto preferencias como salud
- Comparar diferentes enfoques: baselines tradicionales, modelos colaborativos y modelos con integración de features de salud
- Evaluar el trade-off entre precisión de recomendación y calidad nutricional
- Desarrollar métricas específicas para medir la "salud" de las recomendaciones

## Dataset

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

## Arquitectura del Sistema

### 1. Modelos Baseline (Utilizados en E1 principalmente)
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

## Métricas de Evaluación

### Métricas de Precisión
- **Recall@k**: Proporción de items relevantes recuperados
- **NDCG@k**: Calidad del ranking normalizada

### Métricas de Salud
- **WHO_health@k**: Promedio de scores WHO en recomendaciones
- **FSA_health@k**: Promedio de scores FSA en recomendaciones

## 🔍 Hallazgos Principales

1. **DeepFM** logra el mejor balance general entre precisión y salud al integrar features nutricionales directamente
2. **Re-ranking** es efectivo pero sacrifica más precisión que la integración directa
3. El parámetro **α = 0.4** ofrece el mejor trade-off para la mayoría de modelos
4. Los scores **WHO** y **FSA** capturan aspectos complementarios de la salud nutricional

## Dataset original

- Dataset MealRec+: (https://github.com/WUT-IDEA/MealRecPlus)

## Equipo

**Proyecto Grupo 2 - Sistemas Recomendadores 2025-2**

- Andrés Mundaca Zuñiga
- Adriana Bastias Mendoza
- Andres Riquelme Le Roy
