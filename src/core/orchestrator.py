# src/core/orchestrator.py
import logging
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio

class TrainingSystemOrchestrator:
    """
    Orquestador principal del sistema de entrenamiento.
    Coordina todos los componentes y controla el flujo de entrenamiento.
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Componentes principales
        self.teacher = None
        self.student = None
        self.evaluator = None
        
        # Estado del sistema
        self.training_state = {
            'current_cycle': 0,
            'total_questions_processed': 0,
            'improvements_detected': 0,
            'training_active': False
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga configuración del sistema"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_logging(self) -> logging.Logger:
        """Configura sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data/logs/training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def initialize_system(self):
        """Inicializa todos los componentes del sistema"""
        self.logger.info("Inicializando sistema de entrenamiento...")
        
        try:
            # Inicializar componentes
            from .teacher_ai import TeacherAIController
            from .student_ai import StudentAIModel
            from ..evaluation.model_evaluator import ModelEvaluator
            
            self.teacher = TeacherAIController(self.config['teacher'])
            self.student = StudentAIModel(self.config['student'])
            self.evaluator = ModelEvaluator(self.config['evaluation'])
            
            # Verificar conexiones
            await self._verify_connections()
            
            self.logger.info("Sistema inicializado correctamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error en inicialización: {e}")
            return False
    
    async def _verify_connections(self):
        """Verifica conexiones con APIs y recursos"""
        # Verificar Teacher AI (OpenAI API)
        await self.teacher.verify_connection()
        
        # Verificar Student AI (modelo local)
        await self.student.verify_model()
        
        # Verificar acceso a almacenamiento
        # await self._verify_storage_access()
    
    async def execute_training_cycle(self) -> Dict[str, Any]:
        """Ejecuta un ciclo completo de entrenamiento"""
        cycle_results = {
            'cycle_number': self.training_state['current_cycle'],
            'questions_processed': 0,
            'correct_answers': 0,
            'improvements_made': 0,
            'performance_metrics': {}
        }
        
        try:
            self.logger.info(f"Iniciando ciclo {self.training_state['current_cycle']}")
            
            # 1. Generar preguntas con Teacher AI
            questions = await self.teacher.generate_questions(
                count=self.config['training']['questions_per_cycle']
            )
            
            # 2. Procesar cada pregunta
            for question in questions:
                # Student responde
                student_answer = await self.student.generate_answer(question)
                
                # Teacher evalúa
                evaluation = await self.teacher.evaluate_response(
                    question, student_answer
                )
                
                cycle_results['questions_processed'] += 1
                
                if evaluation['is_correct']:
                    cycle_results['correct_answers'] += 1
                else:
                    # Aplicar corrección
                    improvement = await self.student.apply_correction(
                        question, student_answer, evaluation['feedback']
                    )
                    if improvement:
                        cycle_results['improvements_made'] += 1
            
            # 3. Actualizar métricas
            cycle_results['performance_metrics'] = await self._calculate_metrics(cycle_results)
            
            # 4. Actualizar estado
            self.training_state['current_cycle'] += 1
            self.training_state['total_questions_processed'] += cycle_results['questions_processed']
            self.training_state['improvements_detected'] += cycle_results['improvements_made']
            
            self.logger.info(f"Ciclo completado: {cycle_results['correct_answers']}/{cycle_results['questions_processed']} correctas")
            
            return cycle_results
            
        except Exception as e:
            self.logger.error(f"Error en ciclo de entrenamiento: {e}")
            raise
    
    async def _calculate_metrics(self, cycle_results: Dict) -> Dict[str, float]:
        """Calcula métricas de rendimiento del ciclo"""
        total_q = cycle_results['questions_processed']
        correct = cycle_results['correct_answers']
        
        return {
            'accuracy': correct / total_q if total_q > 0 else 0.0,
            'improvement_rate': cycle_results['improvements_made'] / total_q if total_q > 0 else 0.0,
            'learning_efficiency': cycle_results['improvements_made'] / max(1, total_q - correct)
        }
    
    async def run_full_training(self, max_cycles: Optional[int] = None) -> Dict[str, Any]:
        """Ejecuta entrenamiento completo con múltiples ciclos"""
        max_cycles = max_cycles or self.config['training']['max_cycles']
        training_results = {
            'cycles_completed': 0,
            'total_performance': {},
            'learning_progression': [],
            'final_evaluation': {}
        }
        
        self.training_state['training_active'] = True
        
        try:
            # Evaluación inicial
            baseline = await self.evaluator.run_baseline_evaluation(self.student)
            training_results['baseline_performance'] = baseline
            
            # Ciclos de entrenamiento
            for cycle in range(max_cycles):
                if not self.training_state['training_active']:
                    break
                
                cycle_result = await self.execute_training_cycle()
                training_results['learning_progression'].append(cycle_result)
                training_results['cycles_completed'] += 1
                
                # Verificar criterios de parada
                if await self._should_stop_training(cycle_result):
                    break
            
            # Evaluación final
            final_eval = await self.evaluator.run_final_evaluation(
                self.student, baseline
            )
            training_results['final_evaluation'] = final_eval
            
            await self._save_training_results(training_results)
            
            return training_results
            
        finally:
            self.training_state['training_active'] = False
    
    async def _should_stop_training(self, cycle_result: Dict) -> bool:
        """Determina si debe detener el entrenamiento"""
        # Criterios de parada
        min_accuracy = self.config['training']['min_accuracy_threshold']
        max_plateau_cycles = self.config['training']['max_plateau_cycles']
        
        # Verificar accuracy mínima alcanzada
        if cycle_result['performance_metrics']['accuracy'] >= min_accuracy:
            self.logger.info(f"Accuracy objetivo alcanzada: {cycle_result['performance_metrics']['accuracy']:.2%}")
            return True
        
        # Verificar plateau en learning (simplificado)
        if len(self.training_state.get('recent_improvements', [])) >= max_plateau_cycles:
            recent_avg = sum(self.training_state['recent_improvements']) / len(self.training_state['recent_improvements'])
            if recent_avg < 0.01:  # Menos del 1% de mejora
                self.logger.info("Plateau detectado - deteniendo entrenamiento")
                return True
        
        return False
    
    async def _save_training_results(self, results: Dict[str, Any]):
        """Guarda resultados del entrenamiento"""
        from ..data_management.json_manager import JSONManager
        
        json_manager = JSONManager(self.config['data']['output_path'])
        await json_manager.save_training_session(results)
        
        self.logger.info("Resultados de entrenamiento guardados")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna estado actual del sistema"""
        return {
            'training_state': self.training_state,
            'components_status': {
                'teacher_ready': self.teacher is not None,
                'student_ready': self.student is not None,
                'evaluator_ready': self.evaluator is not None
            },
            'config_loaded': bool(self.config)
        }
    
    async def finalize_training(self):
        """Finaliza el proceso de entrenamiento y limpia recursos"""
        self.logger.info("Finalizando sistema de entrenamiento...")
        
        # Guardar estado final
        if self.student:
            await self.student.save_model()
        
        # Limpiar recursos
        if self.teacher:
            await self.teacher.cleanup()
        
        self.training_state['training_active'] = False
        self.logger.info("Sistema finalizado correctamente")