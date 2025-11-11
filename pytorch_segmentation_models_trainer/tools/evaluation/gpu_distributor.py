# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-10-15
        git sha              : $Format:%H$
        copyright            : (C) 2025 by Philipe Borba - Cartographic Engineer
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ****
"""

import logging
import subprocess
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class GPUDistributor:
    """
    Distribui experimentos entre GPUs disponíveis.
    
    Estratégias suportadas:
    - one_experiment_per_gpu: Um experimento por GPU (round-robin)
    - sequential: Executa sequencialmente (sem paralelização)
    
    Features:
    - Auto-detecção de GPUs
    - Lista customizada de GPUs
    - Verificação de memória disponível
    - Balanceamento de carga
    """
    
    def __init__(self, config: DictConfig):
        """
        Args:
            config: DictConfig do pipeline com pipeline_options.parallel_inference
        """
        self.config = config
        self.parallel_config = config.pipeline_options.parallel_inference
        self.strategy = self.parallel_config.strategy
        
        # Detectar ou usar GPUs configuradas
        self.available_gpus = self._detect_gpus()
        
        logger.info(f"GPUDistributor initialized")
        logger.info(f"  Strategy: {self.strategy}")
        logger.info(f"  Available GPUs: {self.available_gpus}")
        
        if len(self.available_gpus) == 0:
            logger.warning("No GPUs detected! Will use CPU.")
    
    def _detect_gpus(self) -> List[int]:
        """
        Detecta GPUs disponíveis ou usa lista configurada.
        
        Returns:
            Lista de IDs de GPUs disponíveis
        """
        # Se GPUs foram especificadas manualmente
        if self.parallel_config.gpus is not None:
            gpus = self.parallel_config.gpus
            logger.info(f"Using manually specified GPUs: {gpus}")
            return gpus
        
        # Auto-detectar GPUs com PyTorch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. No GPUs detected.")
            return []
        
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            logger.warning("No CUDA devices found.")
            return []
        
        # Listar GPUs e verificar memória
        available_gpus = []
        for gpu_id in range(num_gpus):
            try:
                # Verificar propriedades da GPU
                props = torch.cuda.get_device_properties(gpu_id)
                total_memory_gb = props.total_memory / (1024**3)
                
                # Verificar memória livre
                torch.cuda.set_device(gpu_id)
                free_memory_gb = (
                    torch.cuda.get_device_properties(gpu_id).total_memory - 
                    torch.cuda.memory_allocated(gpu_id)
                ) / (1024**3)
                
                logger.info(
                    f"GPU {gpu_id}: {props.name} | "
                    f"Total: {total_memory_gb:.1f}GB | "
                    f"Free: {free_memory_gb:.1f}GB"
                )
                
                # Adicionar se tiver pelo menos 1GB livre
                if free_memory_gb > 1.0:
                    available_gpus.append(gpu_id)
                else:
                    logger.warning(
                        f"GPU {gpu_id} has only {free_memory_gb:.1f}GB free. Skipping."
                    )
                    
            except Exception as e:
                logger.error(f"Error checking GPU {gpu_id}: {e}")
                continue
        
        if len(available_gpus) == 0:
            logger.warning("No GPUs with sufficient memory. Will use CPU.")
        
        return available_gpus
    
    def assign_experiments(
        self, 
        experiments: List[DictConfig]
    ) -> Dict[int, List[DictConfig]]:
        """
        Distribui experimentos entre GPUs.
        
        Estratégias:
        - one_experiment_per_gpu: Round-robin simples
        - sequential: Todos na GPU 0 (ou CPU)
        
        Args:
            experiments: Lista de configs de experimentos
            
        Returns:
            Dict mapeando GPU_ID -> lista de experimentos
            GPU_ID = -1 significa CPU
        """
        if not self.parallel_config.enabled:
            logger.info("Parallel inference disabled. Using sequential execution.")
            return self._assign_sequential(experiments)
        
        if self.strategy == "one_experiment_per_gpu":
            return self._assign_one_per_gpu(experiments)
        
        elif self.strategy == "sequential":
            return self._assign_sequential(experiments)
        
        else:
            logger.warning(
                f"Unknown strategy: {self.strategy}. "
                f"Falling back to sequential."
            )
            return self._assign_sequential(experiments)
    
    def _assign_one_per_gpu(
        self, 
        experiments: List[DictConfig]
    ) -> Dict[int, List[DictConfig]]:
        """
        Distribui experimentos em round-robin entre GPUs.
        
        Exemplo com 3 GPUs e 7 experimentos:
        GPU 0: exp0, exp3, exp6
        GPU 1: exp1, exp4
        GPU 2: exp2, exp5
        
        Args:
            experiments: Lista de experimentos
            
        Returns:
            Dict {gpu_id: [experiments]}
        """
        logger.info("Assigning experiments using one_experiment_per_gpu strategy")
        
        if len(self.available_gpus) == 0:
            logger.warning("No GPUs available. Using CPU for all experiments.")
            return {-1: experiments}  # -1 = CPU
        
        # Inicializar assignments
        assignments = {gpu_id: [] for gpu_id in self.available_gpus}
        
        # Distribuir em round-robin
        for idx, exp in enumerate(experiments):
            gpu_id = self.available_gpus[idx % len(self.available_gpus)]
            assignments[gpu_id].append(exp)
            
            logger.info(
                f"Assigned experiment '{exp.name}' to GPU {gpu_id}"
            )
        
        # Log summary
        logger.info("Assignment summary:")
        for gpu_id, exps in assignments.items():
            exp_names = [exp.name for exp in exps]
            logger.info(f"  GPU {gpu_id}: {len(exps)} experiments - {exp_names}")
        
        return assignments
    
    def _assign_sequential(
        self, 
        experiments: List[DictConfig]
    ) -> Dict[int, List[DictConfig]]:
        """
        Execução sequencial - todos experimentos na mesma GPU/CPU.
        
        Args:
            experiments: Lista de experimentos
            
        Returns:
            Dict {gpu_id: [all experiments]}
        """
        logger.info("Assigning experiments using sequential strategy")
        
        # Usar primeira GPU disponível, ou CPU se nenhuma
        device_id = self.available_gpus[0] if self.available_gpus else -1
        
        device_name = f"GPU {device_id}" if device_id >= 0 else "CPU"
        logger.info(
            f"All {len(experiments)} experiments will run on {device_name}"
        )
        
        return {device_id: experiments}
    
    def get_device_for_experiment(
        self, 
        experiment_name: str,
        assignments: Dict[int, List[DictConfig]]
    ) -> int:
        """
        Retorna o device ID para um experimento específico.
        
        Args:
            experiment_name: Nome do experimento
            assignments: Dict de assignments
            
        Returns:
            GPU ID ou -1 para CPU
        """
        for gpu_id, exps in assignments.items():
            for exp in exps:
                if exp.name == experiment_name:
                    return gpu_id
        
        # Não encontrado - usar primeira GPU ou CPU
        return self.available_gpus[0] if self.available_gpus else -1
    
    def check_gpu_availability(self, gpu_id: int) -> bool:
        """
        Verifica se uma GPU está disponível e acessível.
        
        Args:
            gpu_id: ID da GPU
            
        Returns:
            True se disponível, False caso contrário
        """
        if gpu_id == -1:  # CPU
            return True
        
        if not torch.cuda.is_available():
            return False
        
        if gpu_id >= torch.cuda.device_count():
            return False
        
        try:
            # Tentar alocar um tensor pequeno
            torch.cuda.set_device(gpu_id)
            _ = torch.zeros(1).cuda()
            return True
        except Exception as e:
            logger.error(f"GPU {gpu_id} not accessible: {e}")
            return False
    
    def get_gpu_memory_info(self, gpu_id: int) -> Dict[str, float]:
        """
        Retorna informações de memória de uma GPU.
        
        Args:
            gpu_id: ID da GPU
            
        Returns:
            Dict com 'total_gb', 'used_gb', 'free_gb'
        """
        if gpu_id == -1:  # CPU
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0}
        
        if not self.check_gpu_availability(gpu_id):
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0}
        
        try:
            torch.cuda.set_device(gpu_id)
            props = torch.cuda.get_device_properties(gpu_id)
            total_gb = props.total_memory / (1024**3)
            used_gb = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            free_gb = total_gb - used_gb
            
            return {
                'total_gb': total_gb,
                'used_gb': used_gb,
                'free_gb': free_gb
            }
        except Exception as e:
            logger.error(f"Error getting memory info for GPU {gpu_id}: {e}")
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0}
    
    def wait_for_gpu_memory(
        self, 
        gpu_id: int, 
        required_gb: float,
        timeout_seconds: int = 300
    ) -> bool:
        """
        Aguarda até que uma GPU tenha memória suficiente disponível.
        
        Args:
            gpu_id: ID da GPU
            required_gb: Memória requerida em GB
            timeout_seconds: Timeout em segundos
            
        Returns:
            True se memória ficou disponível, False se timeout
        """
        import time
        
        logger.info(
            f"Waiting for GPU {gpu_id} to have {required_gb:.1f}GB free..."
        )
        
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            mem_info = self.get_gpu_memory_info(gpu_id)
            
            if mem_info['free_gb'] >= required_gb:
                logger.info(
                    f"GPU {gpu_id} now has {mem_info['free_gb']:.1f}GB free"
                )
                return True
            
            # Aguardar 5 segundos antes de verificar novamente
            time.sleep(5)
        
        logger.error(
            f"Timeout waiting for GPU {gpu_id} to have {required_gb:.1f}GB free"
        )
        return False
    
    def get_optimal_batch_size(
        self, 
        gpu_id: int,
        base_batch_size: int = 16,
        memory_per_sample_mb: float = 100.0
    ) -> int:
        """
        Calcula batch size ótimo baseado na memória disponível.
        
        Args:
            gpu_id: ID da GPU
            base_batch_size: Batch size base/padrão
            memory_per_sample_mb: Memória aproximada por sample em MB
            
        Returns:
            Batch size recomendado
        """
        if gpu_id == -1:  # CPU
            return base_batch_size
        
        mem_info = self.get_gpu_memory_info(gpu_id)
        
        if mem_info['free_gb'] == 0:
            return base_batch_size
        
        # Usar 80% da memória livre
        usable_memory_mb = mem_info['free_gb'] * 1024 * 0.8
        
        # Calcular batch size ótimo
        optimal_batch = int(usable_memory_mb / memory_per_sample_mb)
        
        # Limitar entre 1 e base_batch_size * 4
        optimal_batch = max(1, min(optimal_batch, base_batch_size * 4))
        
        logger.debug(
            f"GPU {gpu_id}: Optimal batch size = {optimal_batch} "
            f"(free memory: {mem_info['free_gb']:.1f}GB)"
        )
        
        return optimal_batch
