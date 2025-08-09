# src/core/teacher_ai.py
import asyncio
import openai
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

class TeacherAIController:
    """
    Controlador del Teacher AI que usa GPT-4 para generar preguntas,
    evaluar respuestas y proporcionar feedback correctivo.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configurar OpenAI
        openai.api_key = config['api_key']
        self.model_name = config.get('model', 'gpt-4')
        
        # Estadísticas
        self.stats = {
            'questions_generated': 0,
            'evaluations_completed': 0,
            'api_calls_made': 0
        }
    
    async def verify_connection(self) -> bool:
        """Verifica conexión con OpenAI API"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello, are you working?"}],
                max_tokens=10
            )
            self.logger.info("Conexión con OpenAI API verificada")
            return True
        except Exception as e:
            self.logger.error(f"Error de conexión con OpenAI: {e}")
            return False
    
    async def generate_questions(self, count: int, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Genera preguntas usando GPT-4"""
        prompt = self._build_question_generation_prompt(count, topic)
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
            
            self.stats['api_calls_made'] += 1
            
            # Parsear respuesta JSON
            questions_data = json.loads(response.choices[0].message.content)
            questions = []
            
            for q_data in questions_data['questions']:
                question = {
                    'id': f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(questions)}",
                    'text': q_data['question'],
                    'expected_answer': q_data.get('expected_answer'),
                    'topic': topic or q_data.get('topic', 'general'),
                    'difficulty': q_data.get('difficulty', 'medium'),
                    'created_at': datetime.now().isoformat()
                }
                questions.append(question)
            
            self.stats['questions_generated'] += len(questions)
            self.logger.info(f"📝 Generadas {len(questions)} preguntas")
            
            return questions
            
        except Exception as e:
            self.logger.error(f"Error generando preguntas: {e}")
            return []
    
    def _build_question_generation_prompt(self, count: int, topic: Optional[str]) -> str:
        """Construye prompt para generación de preguntas"""
        topic_instruction = f"sobre el tema '{topic}'" if topic else "de conocimiento general"
        
        return f"""
        Genera {count} preguntas {topic_instruction} para evaluar el conocimiento de un modelo de IA.
        
        Las preguntas deben ser:
        - Claras y específicas
        - Variadas en dificultad (fácil, medio, difícil)
        - Que permitan respuestas evaluables objetivamente
        
        Formato de respuesta (JSON):
        {{
            "questions": [
                {{
                    "question": "¿Pregunta aquí?",
                    "expected_answer": "Respuesta esperada",
                    "topic": "tema",
                    "difficulty": "easy|medium|hard"
                }}
            ]
        }}
        
        Responde solo con el JSON, sin texto adicional.
        """
    
    async def evaluate_response(self, question: Dict[str, Any], student_answer: str) -> Dict[str, Any]:
        """Evalúa respuesta del estudiante usando GPT-4"""
        prompt = self._build_evaluation_prompt(question, student_answer)
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1  # Más determinístico para evaluación
            )
            
            self.stats['api_calls_made'] += 1
            
            # Parsear evaluación
            eval_data = json.loads(response.choices[0].message.content)
            
            evaluation = {
                'question_id': question['id'],
                'is_correct': eval_data['is_correct'],
                'score': eval_data['score'],
                'feedback': eval_data['feedback'],
                'correct_answer': eval_data.get('correct_answer'),
                'areas_to_improve': eval_data.get('areas_to_improve', []),
                'evaluated_at': datetime.now().isoformat()
            }
            
            self.stats['evaluations_completed'] += 1
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluando respuesta: {e}")
            return {
                'question_id': question['id'],
                'is_correct': False,
                'score': 0.0,
                'feedback': f"Error en evaluación: {str(e)}",
                'evaluated_at': datetime.now().isoformat()
            }
    
    def _build_evaluation_prompt(self, question: Dict[str, Any], student_answer: str) -> str:
        """Construye prompt para evaluación de respuestas"""
        return f"""
        Evalúa la siguiente respuesta de un estudiante IA:
        
        PREGUNTA: {question['text']}
        RESPUESTA DEL ESTUDIANTE: {student_answer}
        RESPUESTA ESPERADA: {question.get('expected_answer', 'No especificada')}
        
        Evalúa considerando:
        - Exactitud de la información
        - Completitud de la respuesta
        - Claridad de la explicación
        
        Formato de respuesta (JSON):
        {{
            "is_correct": true/false,
            "score": 0.0-1.0,
            "feedback": "Explicación detallada de la evaluación",
            "correct_answer": "Respuesta correcta si la del estudiante es incorrecta",
            "areas_to_improve": ["área1", "área2"]
        }}
        
        Responde solo con el JSON, sin texto adicional.
        """
    
    async def generate_correction_feedback(self, question: Dict, wrong_answer: str, correct_answer: str) -> str:
        """Genera feedback correctivo detallado"""
        prompt = f"""
        Genera feedback correctivo para ayudar a un modelo IA a aprender:
        
        PREGUNTA: {question['text']}
        RESPUESTA INCORRECTA: {wrong_answer}
        RESPUESTA CORRECTA: {correct_answer}
        
        Proporciona:
        1. Explicación clara de por qué la respuesta es incorrecta
        2. Pasos para llegar a la respuesta correcta
        3. Conceptos clave que debe recordar
        
        Sé constructivo y educativo.
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            self.stats['api_calls_made'] += 1
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generando feedback: {e}")
            return "Error al generar feedback correctivo."
    
    def get_teaching_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del teacher"""
        return {
            **self.stats,
            'api_efficiency': self.stats['evaluations_completed'] / max(1, self.stats['api_calls_made'])
        }
    
    async def cleanup(self):
        """Limpia recursos del teacher"""
        self.logger.info("Limpiando recursos del Teacher AI")
        # Guardar estadísticas finales si es necesario
