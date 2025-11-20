# **🌾 Contador y Clasificador de Semillas de Trigo con AI 🔮**
---
Link: https://seedcounter.streamlit.app/
---
Esta aplicación web, desarrollada con **Streamlit** y **Python**, automatiza la detección y el conteo de semillas y espigas de trigo en imágenes. Conecta con un modelo de Visión por Computador alojado en **Roboflow** para realizar la inferencia en la nube.

**¿Cómo funciona?**
El usuario sube una imagen y la app consulta la API de Roboflow. Los resultados se procesan localmente usando **OpenCV**, permitiendo filtrar las detecciones en tiempo real mediante sliders de **Confianza** y **Superposición (IoU)** sin volver a consumir créditos de la API. Finalmente, muestra la imagen analizada junto con una tabla estadística del conteo por clases.
