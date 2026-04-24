# Práctica 11: ML para Reposicionamiento de Fármacos

Utiliza Regresión Simbólica con PySR y descriptores moleculares de RDKit para predecir la actividad biológica de fármacos de la FDA.

## Instrucciones de uso

Asegúrate de tener `Practica11.py`, el archivo CSV de entrenamiento (`DOWNLOAD-...csv`) y el listado de la FDA (`Launched 30may25.txt`) en tu carpeta. *Nota: Requiere tener el lenguaje Julia instalado en tu sistema.* Abre tu terminal y corre estos comandos en orden para preparar el entorno, instalar lo necesario y ejecutar el script:

**En Ubuntu (WSL / Linux):**
```bash
# Crear entorno, activar, instalar dependencias, configurar PySR y ejecutar
python3 -m venv bioenv
source bioenv/bin/activate
pip install pandas numpy rdkit scikit-learn pysr
python3 -c "import pysr; pysr.install()"
python3 Practica11.py
```

**En Windows (CMD):**
```cmd
# Crear entorno, activar, instalar dependencias, configurar PySR y ejecutar
python -m venv bioenv
bioenv\Scripts\activate
pip install pandas numpy rdkit scikit-learn pysr
python -c "import pysr; pysr.install()"
python Practica11.py
```
