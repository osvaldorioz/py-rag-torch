
El programa es un sistema RAG usando C++, pybind11 y PyTorch. El modelo utilizado en el programa es **`all-MiniLM-L6-v2`** es un **modelo Transformer** optimizado para tareas de **embeddings de texto** (representaciones vectoriales) y **búsqueda semántica**.  

### 🔹 **Características principales:**  
- **Arquitectura:** Basado en **MiniLM** (una versión reducida de BERT).  
- **Número de capas:** 6 (`L6` significa que tiene 6 capas de atención).  
- **Dimensión del embedding:** 384 dimensiones.  
- **Entrenado por:** `sentence-transformers` (de Hugging Face y la Universidad de Mannheim).  
- **Optimización:** Diseñado para ser rápido y eficiente, sin perder calidad en tareas de similitud de texto.  
- **Uso común:**  
  - Búsqueda semántica 🔍  
  - Comparación de frases 📏  
  - Embeddings para NLP 📊  

