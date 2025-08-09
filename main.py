# main.py
import asyncio
import argparse
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

from src.core.orchestrator import TrainingSystemOrchestrator

async def main():
    """Función principal de la aplicación"""
    parser = argparse.ArgumentParser(description='AI Training System')
    parser.add_argument('--config', default='configs/config.yaml', 
                       help='Ruta al archivo de configuración')
    parser.add_argument('--cycles', type=int, default=None,
                       help='Número máximo de ciclos de entrenamiento')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'status'], 
                       default='train', help='Modo de operación')
    
    args = parser.parse_args()
    
    # Cargar variables de entorno
    load_dotenv('configs/api_keys.env')
    
    # Verificar que existen las claves necesarias
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY no encontrada en variables de entorno")
        return
    
    # Crear directorios si no existen
    Path('data/logs').mkdir(parents=True, exist_ok=True)
    Path('data/output').mkdir(parents=True, exist_ok=True)
    
    # Inicializar orquestador
    orchestrator = TrainingSystemOrchestrator(args.config)
    
    try:
        # Inicializar sistema
        if not await orchestrator.initialize_system():
            print("Error en inicialización del sistema")
            return
        
        if args.mode == 'status':
            # Mostrar estado del sistema
            status = orchestrator.get_system_status()
            print("Estado del Sistema:")
            for key, value in status.items():
                print(f"  {key}: {value}")
                
        elif args.mode == 'evaluate':
            # Solo ejecutar evaluación
            results = await orchestrator.evaluator.run_baseline_evaluation(
                orchestrator.student
            )
            print("Resultados de evaluación:")
            print(f"Accuracy: {results.get('accuracy', 0):.2%}")
            
        elif args.mode == 'train':
            # Ejecutar entrenamiento completo
            print("Iniciando entrenamiento...")
            results = await orchestrator.run_full_training(args.cycles)
            
            print("\n🏁 Entrenamiento completado!")
            print(f"  Ciclos completados: {results['cycles_completed']}")
            print(f"  Accuracy final: {results['final_evaluation'].get('accuracy', 0):.2%}")
            print(f"  Mejora total: {results['final_evaluation'].get('improvement', 0):.2%}")
    
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por usuario")
    except Exception as e:
        print(f"Error durante ejecución: {e}")
        logging.exception("Error completo:")
    finally:
        # Limpiar recursos
        await orchestrator.finalize_training()

if __name__ == "__main__":
    asyncio.run(main())
