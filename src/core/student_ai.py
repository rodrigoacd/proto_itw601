# src/core/student_ai.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class StudentAIModel:
    """
    Modelo de IA estudiante que aprende de las correcciones del Teacher.
    Usa modelos pequeños como TinyLlama o Phi para ser eficiente.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.model_name = config.get('model_name', 'microsoft/phi-1_5')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Componentes del modelo
        self.tokenizer = None
        self.model = None
        
        # Historial de aprendizaje
        self.learning_history = []
        self.correction_memory = []
        
        # Métricas
        self.performance_metrics = {
            'questions_answered': 0,
            'corrections_applied': 0,
            'improvement_score': 0.0
        }
    
    async def verify_model(self) -> bool:
        """Verifica y carga el modelo"""
        try:
            self.logger.info(f"🤖 Cargando modelo: {self.model_name}")
            
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configurar pad token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                trust_remote_code=True,
                device_map='auto' if self.device.type == 'cuda' else None
            )
            
            # Mover a dispositivo si es CPU
            if self.device.type == 'cpu':
                self.model.to(self.device)
            
            self.model.eval()
            
            self.logger.info("Modelo cargado correctamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {e}")
            return False
    
    async def generate_answer(self, question: Dict[str, Any]) -> str:
        """Genera respuesta a una pregunta"""
        try:
            # Construir prompt
            prompt = self._build_answer_prompt(question)
            
            # Tokenizar
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Generar respuesta
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get('max_answer_tokens', 150),
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decodificar respuesta
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_response[len(prompt):].strip()
            
            # Registrar en métricas
            self.performance_metrics['questions_answered'] += 1
            
            # Guardar en historial
            self.learning_history.append({
                'question_id': question['id'],
                'question': question['text'],
                'answer': answer,
                'timestamp': datetime.now().isoformat()
            })
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generando respuesta: {e}")
            return "Lo siento, no puedo responder esta pregunta en este momento."
    
    def _build_answer_prompt(self, question: Dict[str, Any]) -> str:
        """Construye prompt para generar respuesta"""
        # Incluir contexto de correcciones previas si es relevante
        context = self._get_relevant_context(question)
        
        prompt = f"""Preg