#!/bin/bash

echo "🚀 Configurando entorno de desarrollo..."

# Verificar Python
python_version=$(python3 --version 2>&1)
echo "🐍 Versión de Python: $python_version"

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔌 Activando entorno virtual..."
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Actualizar pip
echo "⬆️ Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install -r requirements.txt

# Instalar proyecto en modo desarrollo
echo "🔧 Instalando proyecto en modo desarrollo..."
pip install -e .[dev]

# Verificar instalación
echo "✅ Verificando instalación..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo "🎉 Entorno configurado correctamente!"
echo "Para activar el entorno en el futuro:"
echo "  source venv/bin/activate  # Linux/Mac"
echo "  venv\\Scripts\\activate    # Windows"
