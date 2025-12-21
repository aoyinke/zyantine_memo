# ============ OpenAI嵌入服务 ============
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import pickle
import json
import os
import sqlite3
from datetime import datetime
import faiss
import hashlib


class OpenAIEmbeddingService:
    """OpenAI嵌入服务封装"""

    def __init__(self, api_key: str, base_url: str = "https://openkey.cloud/v1",
                 model: str = "text-embedding-3-small", default_dimensions: int = 256,
                 max_retries: int = 3, request_timeout: int = 30):
        """
        初始化OpenAI嵌入服务

        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 嵌入模型名称
            default_dimensions: 默认维度
            max_retries: 最大重试次数
            request_timeout: 请求超时时间
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.default_dimensions = default_dimensions
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        # 初始化客户端
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            print(f"[嵌入服务] OpenAI客户端初始化成功，模型: {model}")
        except ImportError:
            print("[嵌入服务] 警告：未安装openai库，请运行: pip install openai")
            self.client = None
        except Exception as e:
            print(f"[嵌入服务] 初始化失败: {str(e)}")
            self.client = None

        # 缓存
        self.embedding_cache = {}
        self.request_count = 0
        self.error_count = 0

    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self.client is not None

    def embed_text(self, text: str, dimensions: Optional[int] = None) -> Optional[List[float]]:
        """
        将文本转换为向量

        Args:
            text: 输入文本
            dimensions: 向量维度（可选）

        Returns:
            向量列表，失败返回None
        """
        if not self.client:
            print("[嵌入服务] 客户端未初始化")
            return None

        if not text or not text.strip():
            print("[嵌入服务] 输入文本为空")
            return None

        # 检查缓存
        cache_key = f"{text}_{dimensions or self.default_dimensions}"
        if cache_key in self.embedding_cache:
            print(f"[嵌入服务] 使用缓存向量")
            return self.embedding_cache[cache_key]

        # 准备参数
        dimensions_to_use = dimensions or self.default_dimensions

        # 限制文本长度（OpenAI API有token限制）
        processed_text = self._preprocess_text(text)

        if not processed_text:
            print("[嵌入服务] 预处理后文本为空")
            return None

        # 重试机制
        for attempt in range(self.max_retries):
            try:
                self.request_count += 1

                # 调用OpenAI嵌入API
                response = self.client.embeddings.create(
                    input=processed_text,
                    model=self.model,
                    dimensions=dimensions_to_use
                )

                # 提取向量
                embedding_vector = response.data[0].embedding

                # 验证向量维度
                if len(embedding_vector) != dimensions_to_use:
                    print(f"[嵌入服务] 警告：向量维度不匹配，预期{dimensions_to_use}，实际{len(embedding_vector)}")
                    # 调整维度
                    embedding_vector = self._adjust_vector_dimension(embedding_vector, dimensions_to_use)

                # 缓存结果
                self.embedding_cache[cache_key] = embedding_vector

                # 限制缓存大小
                if len(self.embedding_cache) > 1000:
                    # 移除最旧的缓存项
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]

                print(f"[嵌入服务] 成功生成向量，维度: {len(embedding_vector)}")
                return embedding_vector

            except Exception as e:
                self.error_count += 1
                print(f"[嵌入服务] 嵌入请求失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")

                if attempt < self.max_retries - 1:
                    # 指数退避重试
                    wait_time = 2 ** attempt
                    print(f"[嵌入服务] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"[嵌入服务] 所有重试失败")

        return None

    def embed_batch(self, texts: List[str], dimensions: Optional[int] = None) -> List[Optional[List[float]]]:
        """
        批量转换文本为向量

        Args:
            texts: 文本列表
            dimensions: 向量维度

        Returns:
            向量列表，失败项为None
        """
        results = []
        for text in texts:
            embedding = self.embed_text(text, dimensions)
            results.append(embedding)
        return results

    def _preprocess_text(self, text: str, max_length: int = 8000) -> str:
        """
        预处理文本

        Args:
            text: 原始文本
            max_length: 最大字符长度

        Returns:
            预处理后的文本
        """
        if not text:
            return ""

        # 1. 移除多余空白
        processed = ' '.join(text.split())

        # 2. 截断到最大长度（字符级别，实际token数会更少）
        if len(processed) > max_length:
            print(f"[嵌入服务] 文本过长，截断到 {max_length} 字符")
            processed = processed[:max_length]

        # 3. 确保文本不为空
        if not processed.strip():
            print("[嵌入服务] 警告：预处理后文本为空")
            processed = "无内容"

        return processed

    def _adjust_vector_dimension(self, vector: List[float], target_dim: int) -> List[float]:
        """调整向量维度"""
        current_dim = len(vector)

        if current_dim == target_dim:
            return vector
        elif current_dim > target_dim:
            # 截断
            return vector[:target_dim]
        else:
            # 填充零
            padding = [0.0] * (target_dim - current_dim)
            return vector + padding

    def get_statistics(self) -> Dict:
        """获取服务统计信息"""
        return {
            "model": self.model,
            "default_dimensions": self.default_dimensions,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "success_rate": ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100,
            "cache_size": len(self.embedding_cache),
            "is_available": self.is_available()
        }

    def clear_cache(self):
        """清除缓存"""
        self.embedding_cache.clear()
        print("[嵌入服务] 缓存已清除")

    def test_connection(self) -> bool:
        """测试连接"""
        if not self.client:
            return False

        try:
            # 简单的测试请求
            test_response = self.client.embeddings.create(
                input="测试连接",
                model=self.model,
                dimensions=self.default_dimensions
            )

            if test_response and test_response.data and test_response.data[0].embedding:
                print(f"[嵌入服务] 连接测试成功，向量维度: {len(test_response.data[0].embedding)}")
                return True
        except Exception as e:
            print(f"[嵌入服务] 连接测试失败: {str(e)}")

        return False


class FaissMemoryVectorStore:
    """基于Faiss的向量记忆存储系统（使用OpenAI嵌入）"""

    def __init__(self, storage_path: str = "./zyantine_memory",
                 embedding_service: OpenAIEmbeddingService = None,
                 dimension: int = 256):
        """
        初始化Faiss向量存储

        Args:
            storage_path: 存储路径
            embedding_service: OpenAI嵌入服务实例
            dimension: 向量维度
        """
        self.storage_path = storage_path
        self.dimension = dimension

        # 确保存储路径存在
        os.makedirs(self.storage_path, exist_ok=True)

        # 初始化嵌入服务
        if embedding_service is None:
            print("[Faiss] 警告：未提供嵌入服务，将创建默认服务")
            # 这里应该从配置中读取API密钥
            api_key = os.getenv("OPENAI_API_KEY_OPENCLOUD", "sk-wiHpoarpNTHaep0t54852a32A75a4d6986108b3f6eF7B7B9")
            base_url = "https://openkey.cloud/v1"

            self.embedding_service = OpenAIEmbeddingService(
                api_key=api_key,
                base_url=base_url,
                model="text-embedding-3-small",
                default_dimensions=dimension
            )
        else:
            self.embedding_service = embedding_service

        # 验证嵌入服务
        if not self.embedding_service.is_available():
            print("[Faiss] 警告：嵌入服务不可用")

        # 路径定义
        self.index_path = os.path.join(storage_path, "faiss_index.bin")
        self.metadata_path = os.path.join(storage_path, "memory_metadata.pkl")
        self.vector_map_path = os.path.join(storage_path, "vector_to_memory.json")

        # 初始化Faiss索引
        self.index = None
        self.memory_metadata = {}  # 内存ID -> 记忆元数据
        self.vector_to_memory = {}  # 向量ID -> 内存ID
        self.next_vector_id = 0
        self.index_is_trained = False  # 跟踪索引是否已训练

        # 加载现有索引和元数据
        self._load_existing_index()

    def _load_existing_index(self):
        """加载现有的Faiss索引和元数据"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                # 加载Faiss索引
                print(f"[Faiss] 加载现有索引: {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                self.index_is_trained = True  # 已保存的索引应该是已训练的

                # 加载元数据
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.memory_metadata = data.get('metadata', {})
                    self.vector_to_memory = data.get('vector_map', {})
                    self.next_vector_id = data.get('next_id', 0)

                print(f"[Faiss] 加载了 {len(self.memory_metadata)} 个记忆，{len(self.vector_to_memory)} 个向量")

                # 验证索引与元数据一致性
                if self.index.ntotal != len(self.vector_to_memory):
                    print(f"[Faiss] 警告：索引向量数({self.index.ntotal})与元数据数({len(self.vector_to_memory)})不一致")

            except Exception as e:
                print(f"[Faiss] 加载现有索引失败: {e}")
                self._create_new_index()
        else:
            print("[Faiss] 创建新的索引")
            self._create_new_index()

    def _create_new_index(self):
        """创建新的Faiss索引"""
        # 使用Flat索引（精确搜索）代替IVF索引，避免训练问题
        # Flat索引：精确但较慢，但不需要训练
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index_is_trained = True  # Flat索引不需要训练，直接可用

        # 如果之后想要使用IVF索引，可以在这里创建但需要先训练
        # 对于小规模数据，Flat索引更合适
        """
        # IVF索引：近似搜索，更快但需要训练
        nlist = min(100, 4096)  # 聚类中心数，不能超过数据量
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        self.index.nprobe = 10  # 搜索时检查的聚类中心数
        self.index_is_trained = False  # 需要训练
        """

        # 初始化元数据
        self.memory_metadata = {}
        self.vector_to_memory = {}
        self.next_vector_id = 0

        print(f"[Faiss] 创建新索引，维度: {self.dimension}, 类型: FlatL2")

    def _train_index_if_needed(self, vectors: np.ndarray):
        """如果需要，训练索引（对于IVF索引）"""
        if not self.index_is_trained and hasattr(self.index, 'is_trained') and not self.index.is_trained:
            # 只有IVF索引需要训练
            if len(vectors) >= self.index.nlist:  # 至少有nlist个向量时训练
                print(f"[Faiss] 训练IVF索引，使用 {len(vectors)} 个向量")
                self.index.train(vectors)
                self.index_is_trained = True
                return True
            else:
                print(f"[Faiss] 警告：向量数量({len(vectors)})不足，无法训练IVF索引（需要至少{self.index.nlist}个）")
                # 如果向量不足，临时使用Flat索引
                print(f"[Faiss] 临时使用Flat索引处理少量数据")
                # 创建临时Flat索引
                temp_index = faiss.IndexFlatL2(self.dimension)
                temp_index.add(vectors)
                # 替换当前索引为Flat索引
                self.index = temp_index
                self.index_is_trained = True
                return False
        return False

    def embed_text(self, text: str) -> np.ndarray:
        """将文本转换为向量（使用OpenAI API）"""
        if not text or not text.strip():
            # 返回零向量
            return np.zeros(self.dimension, dtype=np.float32)

        # 使用OpenAI嵌入服务
        embedding_list = self.embedding_service.embed_text(text, self.dimension)

        if embedding_list is None:
            print(f"[Faiss] 文本嵌入失败，使用随机向量: {text[:50]}...")
            # 返回随机向量作为后备
            return np.random.randn(self.dimension).astype(np.float32)

        # 转换为numpy数组
        embedding_array = np.array(embedding_list, dtype=np.float32)

        # 确保维度正确
        if len(embedding_array) != self.dimension:
            print(f"[Faiss] 警告：嵌入维度不匹配，预期{self.dimension}，实际{len(embedding_array)}")
            # 调整维度
            if len(embedding_array) > self.dimension:
                embedding_array = embedding_array[:self.dimension]
            else:
                padding = np.zeros(self.dimension - len(embedding_array), dtype=np.float32)
                embedding_array = np.concatenate([embedding_array, padding])

        return embedding_array

    def add_memory(self, memory_id: str, text: str, metadata: Dict,
                   vector: Optional[np.ndarray] = None) -> int:
        """
        添加记忆到向量索引

        Args:
            memory_id: 记忆的唯一ID
            text: 记忆文本内容
            metadata: 记忆元数据
            vector: 可选的预计算向量（如果为None则从文本生成）

        Returns:
            向量ID
        """
        # 生成向量ID
        vector_id = self.next_vector_id
        self.next_vector_id += 1

        # 计算或使用提供的向量
        if vector is None:
            vector = self.embed_text(text)
        else:
            # 确保向量维度正确
            if len(vector) != self.dimension:
                vector = self._adjust_vector_dimension(vector)

        # 将向量添加到索引
        vector_reshaped = vector.reshape(1, -1)

        # 检查索引是否需要训练（仅对IVF索引）
        if not self.index_is_trained and hasattr(self.index, 'is_trained') and not self.index.is_trained:
            # 对于IVF索引，需要先训练
            self._train_index_if_needed(vector_reshaped)

        # 添加到索引
        try:
            self.index.add(vector_reshaped)
        except Exception as e:
            print(f"[Faiss] 添加向量到索引失败: {e}")
            # 如果添加失败，可能是索引需要训练但训练失败
            # 创建新的Flat索引并添加向量
            print(f"[Faiss] 创建新的Flat索引并重新添加")
            old_index = self.index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index_is_trained = True

            # 如果旧索引中有向量，尝试迁移
            if hasattr(old_index, 'ntotal') and old_index.ntotal > 0:
                try:
                    # 获取旧索引中的所有向量
                    old_vectors = self._extract_all_vectors_from_index(old_index)
                    if old_vectors is not None and len(old_vectors) > 0:
                        self.index.add(old_vectors)
                        print(f"[Faiss] 迁移了 {len(old_vectors)} 个向量到新索引")
                except Exception as e2:
                    print(f"[Faiss] 迁移旧向量失败: {e2}")

            # 重新添加当前向量
            self.index.add(vector_reshaped)

        # 存储元数据
        self.memory_metadata[memory_id] = {
            'vector_id': vector_id,
            'text': text,
            'metadata': metadata,
            'added_at': datetime.now().isoformat(),
            'access_count': 0,
            'embedding_source': 'openai' if vector is None else 'precomputed'
        }

        # 存储向量ID到内存ID的映射
        self.vector_to_memory[vector_id] = memory_id

        print(f"[Faiss] 添加记忆 {memory_id}，向量ID: {vector_id}，维度: {len(vector)}")

        return vector_id

    def _extract_all_vectors_from_index(self, index):
        """从索引中提取所有向量（对于Flat索引）"""
        try:
            # 对于Flat索引，可以尝试重建
            if hasattr(index, 'reconstruct_n'):
                # 尝试重构向量
                ntotal = index.ntotal
                if ntotal > 0:
                    vectors = np.zeros((ntotal, self.dimension), dtype=np.float32)
                    for i in range(ntotal):
                        vectors[i] = index.reconstruct(i)
                    return vectors
        except Exception as e:
            print(f"[Faiss] 提取向量失败: {e}")
        return None

    def _adjust_vector_dimension(self, vector: np.ndarray) -> np.ndarray:
        """调整向量维度"""
        if len(vector) > self.dimension:
            return vector[:self.dimension].astype(np.float32)
        elif len(vector) < self.dimension:
            padding = np.zeros(self.dimension - len(vector), dtype=np.float32)
            return np.concatenate([vector, padding]).astype(np.float32)
        else:
            return vector.astype(np.float32)

    def search_similar(self, query_text: str, top_k: int = 5,
                       threshold: float = 0.5) -> List[Dict]:
        """
        搜索相似的记忆

        Args:
            query_text: 查询文本
            top_k: 返回的最相似数量
            threshold: 相似度阈值（0-1）

        Returns:
            相似记忆列表，按相似度排序
        """
        if self.index.ntotal == 0:
            return []

        # 检查索引是否需要训练（对于IVF索引）
        if not self.index_is_trained and hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("[Faiss] 警告：索引未训练，无法搜索")
            return []

        # 将查询文本转换为向量
        query_vector = self.embed_text(query_text)
        query_vector_reshaped = query_vector.reshape(1, -1)

        # 搜索最相似的向量
        try:
            distances, indices = self.index.search(query_vector_reshaped, min(top_k, self.index.ntotal))
        except Exception as e:
            print(f"[Faiss] 搜索失败: {e}")
            return []

        results = []
        for i in range(len(indices[0])):
            vector_id = indices[0][i]
            distance = distances[0][i]

            # 转换距离为相似度分数（0-1）
            similarity = 1.0 / (1.0 + distance)

            if similarity < threshold:
                continue

            # 获取对应的记忆ID
            memory_id = self.vector_to_memory.get(vector_id)
            if not memory_id or memory_id not in self.memory_metadata:
                continue

            # 获取记忆元数据
            memory_info = self.memory_metadata[memory_id]

            # 更新访问计数
            memory_info['access_count'] += 1

            results.append({
                'memory_id': memory_id,
                'vector_id': vector_id,
                'similarity': float(similarity),
                'distance': float(distance),
                'text': memory_info['text'],
                'metadata': memory_info['metadata'],
                'access_count': memory_info['access_count']
            })

        # 按相似度降序排序
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results

    def search_by_metadata(self, filters: Dict) -> List[Dict]:
        """
        根据元数据过滤搜索记忆

        Args:
            filters: 元数据过滤条件，如 {"tags": "成就", "source": "user"}

        Returns:
            匹配的记忆列表
        """
        results = []

        for memory_id, memory_info in self.memory_metadata.items():
            metadata = memory_info['metadata']

            # 检查所有过滤条件是否满足
            match = True
            for key, value in filters.items():
                if key not in metadata:
                    match = False
                    break

                metadata_value = metadata[key]

                # 支持不同的匹配方式
                if isinstance(value, list):
                    if not any(v in metadata_value for v in value):
                        match = False
                        break
                elif isinstance(value, str) and isinstance(metadata_value, list):
                    if value not in metadata_value:
                        match = False
                        break
                elif metadata_value != value:
                    match = False
                    break

            if match:
                results.append({
                    'memory_id': memory_id,
                    'vector_id': memory_info['vector_id'],
                    'text': memory_info['text'],
                    'metadata': metadata,
                    'access_count': memory_info['access_count']
                })

        return results

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """根据ID获取记忆"""
        if memory_id in self.memory_metadata:
            memory_info = self.memory_metadata[memory_id]
            memory_info['access_count'] += 1

            return {
                'memory_id': memory_id,
                'vector_id': memory_info['vector_id'],
                'text': memory_info['text'],
                'metadata': memory_info['metadata'],
                'access_count': memory_info['access_count']
            }

        return None

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆（注意：Faiss不支持直接删除，标记为删除）"""
        if memory_id in self.memory_metadata:
            # 标记为删除（实际删除需要重建索引）
            self.memory_metadata[memory_id]['deleted'] = True
            print(f"[Faiss] 记忆 {memory_id} 标记为删除")
            return True

        return False

    def update_memory(self, memory_id: str, text: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> bool:
        """更新记忆"""
        if memory_id not in self.memory_metadata:
            return False

        memory_info = self.memory_metadata[memory_id]

        if text is not None:
            memory_info['text'] = text
            # 重新计算向量
            new_vector = self.embed_text(text)
            # 更新向量索引（需要重建索引）
            memory_info['needs_reindex'] = True

        if metadata is not None:
            # 合并元数据
            if 'metadata' in memory_info:
                memory_info['metadata'].update(metadata)
            else:
                memory_info['metadata'] = metadata

        memory_info['updated_at'] = datetime.now().isoformat()

        return True

    def save(self):
        """保存索引和元数据到磁盘"""
        # 确保目录存在
        os.makedirs(self.storage_path, exist_ok=True)

        # 保存Faiss索引
        faiss.write_index(self.index, self.index_path)

        # 保存元数据
        metadata_data = {
            'metadata': self.memory_metadata,
            'vector_map': self.vector_to_memory,
            'next_id': self.next_vector_id,
            'saved_at': datetime.now().isoformat(),
            'total_memories': len(self.memory_metadata),
            'dimension': self.dimension,
            'model': self.embedding_service.model
        }

        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata_data, f)

        # 保存向量映射（JSON格式，便于查看）
        vector_map_json = {
            vector_id: memory_id
            for vector_id, memory_id in self.vector_to_memory.items()
        }

        with open(self.vector_map_path, 'w', encoding='utf-8') as f:
            json.dump(vector_map_json, f, ensure_ascii=False, indent=2)

        print(f"[Faiss] 保存索引和元数据，共 {len(self.memory_metadata)} 个记忆")

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_memories = len(self.memory_metadata)

        # 计算各类标签统计
        tag_counts = {}
        source_counts = {}

        for memory_info in self.memory_metadata.values():
            metadata = memory_info.get('metadata', {})

            # 标签统计
            tags = metadata.get('tags', [])
            if isinstance(tags, list):
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # 来源统计
            source = metadata.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        # 计算平均访问次数
        total_access = sum(m['access_count'] for m in self.memory_metadata.values())
        avg_access = total_access / max(total_memories, 1)

        # 获取索引类型
        index_type = "FlatL2"
        if hasattr(self.index, 'is_trained'):
            index_type = "IVFFlat" if hasattr(self.index, 'nlist') else "FlatL2"

        return {
            'total_memories': total_memories,
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': index_type,
            'index_is_trained': self.index_is_trained,
            'embedding_model': self.embedding_service.model,
            'tag_counts': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'source_counts': source_counts,
            'total_access_count': total_access,
            'average_access_count': round(avg_access, 2),
            'most_accessed': self._get_most_accessed_memories(5)
        }

    def _get_most_accessed_memories(self, top_n: int = 5) -> List[Dict]:
        """获取访问次数最多的记忆"""
        memories_with_access = [
            {
                'memory_id': memory_id,
                'access_count': info['access_count'],
                'text': info['text'][:100] + '...' if len(info['text']) > 100 else info['text'],
                'added_at': info.get('added_at', 'unknown')
            }
            for memory_id, info in self.memory_metadata.items()
        ]

        memories_with_access.sort(key=lambda x: x['access_count'], reverse=True)
        return memories_with_access[:top_n]

    def rebuild_index(self):
        """重建索引（用于删除或大量更新后的重建）"""
        print(f"[Faiss] 开始重建索引...")

        # 收集未删除的记忆
        valid_memories = []
        valid_vectors = []

        for memory_id, memory_info in self.memory_metadata.items():
            if memory_info.get('deleted', False):
                continue

            # 重新计算向量（如果需要）
            if memory_info.get('needs_reindex', False):
                vector = self.embed_text(memory_info['text'])
                memory_info['needs_reindex'] = False
            else:
                # 从索引中获取现有向量
                vector_id = memory_info['vector_id']
                if vector_id in self.vector_to_memory:
                    # 获取向量（需要从索引中提取）
                    vector = self._extract_vector(vector_id)
                else:
                    vector = self.embed_text(memory_info['text'])

            valid_memories.append((memory_id, memory_info))
            valid_vectors.append(vector)

        # 创建新索引
        self._create_new_index()

        # 重新添加所有有效记忆
        for (memory_id, memory_info), vector in zip(valid_memories, valid_vectors):
            # 重置向量ID映射
            vector_id = self.next_vector_id
            self.next_vector_id += 1

            memory_info['vector_id'] = vector_id
            self.vector_to_memory[vector_id] = memory_id

            # 添加到新索引
            vector_reshaped = vector.reshape(1, -1)
            self.index.add(vector_reshaped)

        print(f"[Faiss] 重建索引完成，保留 {len(valid_memories)} 个记忆")

    def _extract_vector(self, vector_id: int) -> np.ndarray:
        """从索引中提取指定向量（Faiss没有直接提取的方法，需要重建）"""
        # 注意：这是简化实现，实际可能需要更复杂的逻辑
        # 对于大量数据，可能需要不同的方法
        return np.random.randn(self.dimension).astype(np.float32)  # 占位实现


class HybridMemorySystem:
    """混合记忆系统：Faiss存储向量（使用OpenAI嵌入），SQLite存储关系数据"""

    def __init__(self, storage_path: str = "./zyantine_memory",
                 embedding_service: Optional[OpenAIEmbeddingService] = None):
        self.storage_path = storage_path

        # 确保存储路径存在
        os.makedirs(self.storage_path, exist_ok=True)

        # 初始化OpenAI嵌入服务
        if embedding_service is None:
            # 创建默认服务（从环境变量或配置）
            api_key = os.getenv("OPENAI_API_KEY_OPENCLOUD", "")
            base_url = os.getenv("OPENAI_BASE_URL", "https://openkey.cloud/v1")

            if api_key:
                embedding_service = OpenAIEmbeddingService(
                    api_key=api_key,
                    base_url=base_url,
                    model="text-embedding-3-small",
                    default_dimensions=256
                )
            else:
                print("[混合记忆] 警告：未提供API密钥，嵌入服务可能不可用")
                embedding_service = None

        # 初始化Faiss向量存储（使用OpenAI嵌入）
        self.vector_store = FaissMemoryVectorStore(storage_path, embedding_service, 256)

        # 初始化SQLite存储
        self.db_path = os.path.join(storage_path, "memory_metadata.db")
        self._init_sqlite()

        # 缓存
        self.conversation_cache = {}
        self.vector_state_cache = {}

    def _init_sqlite(self):
        """初始化SQLite数据库"""
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 对话历史表（不需要向量检索的文本数据）
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS conversations
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               TEXT
                               NOT
                               NULL,
                               user_input
                               TEXT
                               NOT
                               NULL,
                               system_response
                               TEXT
                               NOT
                               NULL,
                               context_analysis
                               TEXT,
                               vector_state
                               TEXT,
                               session_id
                               TEXT
                               DEFAULT
                               'default',
                               created_at
                               TIMESTAMP
                               DEFAULT
                               CURRENT_TIMESTAMP
                           )
                           ''')

            # 向量状态历史表
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS vector_history
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               TEXT
                               NOT
                               NULL,
                               TR
                               REAL
                               NOT
                               NULL,
                               CS
                               REAL
                               NOT
                               NULL,
                               SA
                               REAL
                               NOT
                               NULL,
                               interaction_type
                               TEXT,
                               session_id
                               TEXT
                               DEFAULT
                               'default',
                               created_at
                               TIMESTAMP
                               DEFAULT
                               CURRENT_TIMESTAMP
                           )
                           ''')

            # 辩证成长记录表
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS growth_records
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               timestamp
                               TEXT
                               NOT
                               NULL,
                               cycle
                               INTEGER
                               NOT
                               NULL,
                               situation_summary
                               TEXT,
                               validation_result
                               TEXT,
                               new_principle
                               TEXT,
                               session_id
                               TEXT
                               DEFAULT
                               'default',
                               created_at
                               TIMESTAMP
                               DEFAULT
                               CURRENT_TIMESTAMP
                           )
                           ''')

            # 记忆关联表（链接Faiss向量ID和其他数据）
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS memory_relations
                           (
                               memory_id
                               TEXT
                               PRIMARY
                               KEY,
                               faiss_vector_id
                               INTEGER,
                               memory_type
                               TEXT,
                               related_ids
                               TEXT,
                               created_at
                               TIMESTAMP
                               DEFAULT
                               CURRENT_TIMESTAMP
                           )
                           ''')

            # 索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_vector_session ON vector_history(session_id)')

            conn.commit()
            conn.close()
            print(f"[混合记忆] SQLite数据库初始化成功: {self.db_path}")
        except sqlite3.Error as e:
            print(f"[混合记忆] SQLite数据库初始化失败: {e}")
            # 尝试创建父目录
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            # 重新尝试连接
            conn = sqlite3.connect(self.db_path)
            conn.close()
            print(f"[混合记忆] 已创建数据库文件: {self.db_path}")

    def add_conversation_memory(self, conversation: Dict, session_id: str = "default"):
        """添加对话记忆到Faiss和SQLite"""
        import json

        # 生成记忆ID
        memory_id = f"conv_{hashlib.md5(str(conversation).encode()).hexdigest()[:12]}"

        # 准备文本用于向量化（结合用户输入和系统响应）
        conversation_text = f"用户: {conversation.get('user_input', '')} | 自衍体: {conversation.get('system_response', '')}"

        # 准备元数据
        metadata = {
            'type': 'conversation',
            'session_id': session_id,
            'timestamp': conversation.get('timestamp', datetime.now().isoformat()),
            'context': conversation.get('context', {}),
            'tags': ['对话', '交互']
        }

        # 添加到Faiss向量存储
        vector_id = self.vector_store.add_memory(memory_id, conversation_text, metadata)

        # 添加到SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           INSERT INTO conversations
                           (timestamp, user_input, system_response, context_analysis, vector_state, session_id)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ''', (
                               conversation.get('timestamp', datetime.now().isoformat()),
                               conversation.get('user_input', ''),
                               conversation.get('system_response', ''),
                               json.dumps(conversation.get('context', {}), ensure_ascii=False),
                               json.dumps(conversation.get('vector_state', {}), ensure_ascii=False),
                               session_id
                           ))

            # 添加记忆关联
            cursor.execute('''
                INSERT OR REPLACE INTO memory_relations 
                (memory_id, faiss_vector_id, memory_type, related_ids)
                VALUES (?, ?, ?, ?)
            ''', (
                memory_id,
                vector_id,
                'conversation',
                json.dumps({'conversation_id': cursor.lastrowid}, ensure_ascii=False)
            ))

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"[混合记忆] 添加对话记录到SQLite失败: {e}")
            # 尝试重新初始化数据库
            self._init_sqlite()

        return memory_id, vector_id

    def add_experience_memory(self, experience: Dict, session_id: str = "default"):
        """添加经历记忆（用户记忆或自衍体记忆）"""
        memory_id = f"exp_{hashlib.md5(str(experience).encode()).hexdigest()[:12]}"

        # 准备文本
        text = f"{experience.get('summary', '')} | {experience.get('content', '')}"

        # 准备元数据
        metadata = {
            'type': 'experience',
            'session_id': session_id,
            'source': experience.get('source', 'user'),
            'emotional_intensity': experience.get('emotional_intensity', 0.5),
            'tags': experience.get('tags', []),
            'strategic_value': experience.get('strategic_value', {}),
            'linked_tool': experience.get('linked_tool'),
            'timestamp': experience.get('timestamp', datetime.now().isoformat())
        }

        # 添加到Faiss
        vector_id = self.vector_store.add_memory(memory_id, text, metadata)

        return memory_id, vector_id

    def add_vector_state_memory(self, vector_state: Dict, interaction_type: str,
                                session_id: str = "default"):
        """添加向量状态记忆"""

        # 保存到SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           INSERT INTO vector_history
                               (timestamp, TR, CS, SA, interaction_type, session_id)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ''', (
                               vector_state.get('timestamp', datetime.now().isoformat()),
                               vector_state.get('TR', 0.5),
                               vector_state.get('CS', 0.5),
                               vector_state.get('SA', 0.5),
                               interaction_type,
                               session_id
                           ))

            # 也可以将向量状态添加到Faiss用于模式识别
            vector_text = f"向量状态: TR={vector_state.get('TR', 0.5):.2f}, CS={vector_state.get('CS', 0.5):.2f}, SA={vector_state.get('SA', 0.5):.2f}, 交互类型: {interaction_type}"

            memory_id = f"vec_{hashlib.md5(vector_text.encode()).hexdigest()[:12]}"

            metadata = {
                'type': 'vector_state',
                'session_id': session_id,
                'interaction_type': interaction_type,
                'TR': vector_state.get('TR', 0.5),
                'CS': vector_state.get('CS', 0.5),
                'SA': vector_state.get('SA', 0.5),
                'tags': ['向量状态', '情感模式']
            }

            vector_id = self.vector_store.add_memory(memory_id, vector_text, metadata)

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"[混合记忆] 添加向量状态到SQLite失败: {e}")
            # 尝试重新初始化数据库
            self._init_sqlite()
            memory_id = None
            vector_id = None

        return memory_id, vector_id

    def find_similar_conversations(self, query_text: str, top_k: int = 5,
                                   session_id: Optional[str] = None) -> List[Dict]:
        """查找相似的对话"""
        # 在Faiss中搜索
        faiss_results = self.vector_store.search_similar(query_text, top_k * 2)  # 多取一些用于过滤

        results = []
        for result in faiss_results:
            metadata = result['metadata']

            # 过滤会话类型
            if metadata.get('type') != 'conversation':
                continue

            # 过滤会话ID（如果需要）
            if session_id and metadata.get('session_id') != session_id:
                continue

            # 从SQLite获取完整对话数据
            full_conversation = self._get_conversation_by_memory_id(result['memory_id'])
            if full_conversation:
                result['full_conversation'] = full_conversation
                results.append(result)

            if len(results) >= top_k:
                break

        return results

    def find_relevant_experiences(self, context: Dict, top_k: int = 5) -> List[Dict]:
        """查找相关的经历记忆"""
        # 构建查询文本
        query_parts = []

        if 'user_input' in context:
            query_parts.append(f"用户说: {context['user_input']}")

        if 'emotion' in context:
            query_parts.append(f"情绪: {context['emotion']}")

        if 'topic' in context:
            query_parts.append(f"话题: {context['topic']}")

        query_text = " ".join(query_parts)

        # 在Faiss中搜索
        faiss_results = self.vector_store.search_similar(query_text, top_k * 3)

        results = []
        for result in faiss_results:
            metadata = result['metadata']

            # 只返回经历类型的记忆
            if metadata.get('type') == 'experience':
                # 添加战略价值评估
                strategic_value = metadata.get('strategic_value', {})
                if strategic_value and isinstance(strategic_value, dict):
                    result['strategic_level'] = strategic_value.get('level', '低')
                    result['strategic_score'] = strategic_value.get('score', 0)

                results.append(result)

            if len(results) >= top_k:
                break

        return results

    def find_pattern_in_vectors(self, window_size: int = 10,
                                session_id: str = "default") -> List[Dict]:
        """在向量历史中寻找模式"""
        import json

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           SELECT timestamp, TR, CS, SA, interaction_type
                           FROM vector_history
                           WHERE session_id = ?
                           ORDER BY timestamp DESC
                               LIMIT ?
                           ''', (session_id, window_size))

            rows = cursor.fetchall()
            conn.close()

            if len(rows) < 3:
                return []

            patterns = []

            # 检测趋势
            tr_values = [row[1] for row in rows]
            cs_values = [row[2] for row in rows]
            sa_values = [row[3] for row in rows]

            # 计算变化趋势
            tr_trend = self._calculate_trend(tr_values)
            cs_trend = self._calculate_trend(cs_values)
            sa_trend = self._calculate_trend(sa_values)

            # 检测模式
            if sa_trend > 0.7 and cs_trend < -0.5:
                patterns.append({
                    'pattern': '压力上升安全感下降',
                    'confidence': min(sa_trend, abs(cs_trend)),
                    'description': '近期压力增加，安全感下降',
                    'suggestion': '需要关注安全感恢复'
                })

            if tr_trend > 0.6 and cs_trend > 0.4:
                patterns.append({
                    'pattern': '积极成长',
                    'confidence': (tr_trend + cs_trend) / 2,
                    'description': '兴奋感和安全感同步增长',
                    'suggestion': '处于积极成长阶段'
                })

            if abs(tr_values[-1] - 0.6) < 0.1 and abs(cs_values[-1] - 0.6) < 0.1 and abs(sa_values[-1] - 0.3) < 0.1:
                patterns.append({
                    'pattern': '理想平衡',
                    'confidence': 0.8,
                    'description': '向量处于理想平衡状态',
                    'suggestion': '保持当前状态'
                })

            return patterns
        except sqlite3.Error as e:
            print(f"[混合记忆] 查询向量模式失败: {e}")
            return []

    def _calculate_trend(self, values: List[float]) -> float:
        """计算数值序列的趋势（-1到1）"""
        if len(values) < 2:
            return 0.0

        # 简单线性趋势
        x = list(range(len(values)))
        y = values

        # 计算斜率
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x_i * x_i for x_i in x)

        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # 归一化到[-1, 1]
        max_slope = 0.1  # 假设最大斜率
        trend = max(-1.0, min(1.0, slope / max_slope))

        return trend

    def _get_conversation_by_memory_id(self, memory_id: str) -> Optional[Dict]:
        """根据记忆ID从SQLite获取完整对话"""
        import json

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           SELECT mr.related_ids,
                                  c.timestamp,
                                  c.user_input,
                                  c.system_response,
                                  c.context_analysis,
                                  c.vector_state
                           FROM memory_relations mr
                                    LEFT JOIN conversations c
                                              ON json_extract(mr.related_ids, '$.conversation_id') = c.id
                           WHERE mr.memory_id = ?
                           ''', (memory_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                related_ids = json.loads(row[0]) if row[0] else {}
                conversation_id = related_ids.get('conversation_id')

                if conversation_id:
                    return {
                        'id': conversation_id,
                        'timestamp': row[1],
                        'user_input': row[2],
                        'system_response': row[3],
                        'context': json.loads(row[4]) if row[4] else {},
                        'vector_state': json.loads(row[5]) if row[5] else {}
                    }
        except sqlite3.Error as e:
            print(f"[混合记忆] 获取对话记录失败: {e}")

        return None

    def get_conversation_history(self, limit: int = 50, session_id: str = "default") -> List[Dict]:
        """获取对话历史"""
        import json

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           SELECT timestamp, user_input, system_response, context_analysis, vector_state
                           FROM conversations
                           WHERE session_id = ?
                           ORDER BY timestamp DESC
                               LIMIT ?
                           ''', (session_id, limit))

            rows = cursor.fetchall()
            conn.close()

            history = []
            for row in rows:
                history.append({
                    'timestamp': row[0],
                    'user_input': row[1],
                    'system_response': row[2],
                    'context': json.loads(row[3]) if row[3] else {},
                    'vector_state': json.loads(row[4]) if row[4] else {}
                })

            # 按时间正序返回
            return list(reversed(history))
        except sqlite3.Error as e:
            print(f"[混合记忆] 获取对话历史失败: {e}")
            return []

    def get_latest_vector_state(self, session_id: str = "default") -> Optional[Dict]:
        """获取最新的向量状态"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           SELECT TR, CS, SA, interaction_type
                           FROM vector_history
                           WHERE session_id = ?
                           ORDER BY timestamp DESC
                               LIMIT 1
                           ''', (session_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'TR': row[0],
                    'CS': row[1],
                    'SA': row[2],
                    'interaction_type': row[3]
                }
        except sqlite3.Error as e:
            print(f"[混合记忆] 获取向量状态失败: {e}")

        return None

    def save(self):
        """保存所有数据"""
        self.vector_store.save()
        print(f"[混合记忆] 所有数据已保存")

    def get_statistics(self, session_id: str = "default") -> Dict:
        """获取统计信息"""
        # Faiss统计
        faiss_stats = self.vector_store.get_statistics()

        # SQLite统计
        conversation_count = 0
        vector_history_count = 0
        growth_records_count = 0

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM conversations WHERE session_id = ?', (session_id,))
            conversation_count = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM vector_history WHERE session_id = ?', (session_id,))
            vector_history_count = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM growth_records WHERE session_id = ?', (session_id,))
            growth_records_count = cursor.fetchone()[0]

            conn.close()
        except sqlite3.Error as e:
            print(f"[混合记忆] 获取统计信息失败: {e}")

        return {
            'faiss_memories': faiss_stats['total_memories'],
            'conversations': conversation_count,
            'vector_history': vector_history_count,
            'growth_records': growth_records_count,
            'vector_dimension': faiss_stats['dimension'],
            'index_type': faiss_stats['index_type'],
            'embedding_model': faiss_stats['embedding_model'],
            'most_common_tags': faiss_stats['tag_counts'],
            'most_accessed_memories': faiss_stats['most_accessed']
        }

    def backup(self, backup_path: Optional[str] = None):
        """备份记忆系统"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.storage_path, f"backup_{timestamp}")

        os.makedirs(backup_path, exist_ok=True)

        try:
            # 备份Faiss索引
            import shutil
            if os.path.exists(self.vector_store.index_path):
                shutil.copy2(self.vector_store.index_path,
                             os.path.join(backup_path, "faiss_index.bin"))
            if os.path.exists(self.vector_store.metadata_path):
                shutil.copy2(self.vector_store.metadata_path,
                             os.path.join(backup_path, "memory_metadata.pkl"))

            # 备份SQLite数据库
            if os.path.exists(self.db_path):
                shutil.copy2(self.db_path, os.path.join(backup_path, "memory_metadata.db"))

            print(f"[混合记忆] 备份已保存到: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"[混合记忆] 备份失败: {e}")
            return None


# ============ 增强的记忆炼金术引擎（使用Faiss） ============
class EnhancedMemoryAlchemyEngine:
    """增强的记忆炼金术引擎，使用Faiss进行向量检索"""

    def __init__(self, user_profile_data: Dict, self_profile_data: Dict,
                 memory_system: HybridMemorySystem, session_id: str = "default"):
        self.memory_system = memory_system
        self.session_id = session_id

        # 语义记忆地图（从Faiss加载）
        self.semantic_memory_map = {}
        self.strategic_tags = []

        # 从现有记忆初始化
        self._initialize_from_existing_memories()

        # 处理初始记忆数据
        self._process_initial_memories(user_profile_data, self_profile_data)

    def _initialize_from_existing_memories(self):
        """从现有记忆初始化"""
        # 获取所有经历类型的记忆
        experiences = self.memory_system.vector_store.search_by_metadata({
            'type': 'experience'
        })

        for exp in experiences:
            memory_id = exp['memory_id']
            self.semantic_memory_map[memory_id] = {
                'event_summary': exp['metadata'].get('summary', '未知事件'),
                'raw_content': exp['text'],
                'tags': exp['metadata'].get('tags', []),
                'source': exp['metadata'].get('source', 'unknown'),
                'emotional_intensity': exp['metadata'].get('emotional_intensity', 0.5),
                'strategic_value': exp['metadata'].get('strategic_value', {}),
                'access_count': exp['access_count']
            }

            # 更新战略标签
            for tag in exp['metadata'].get('tags', []):
                if tag not in self.strategic_tags:
                    self.strategic_tags.append(tag)

        print(f"[记忆引擎] 从Faiss加载了 {len(self.semantic_memory_map)} 个经历记忆")

    def _process_initial_memories(self, user_profile_data: Dict, self_profile_data: Dict):
        """处理初始记忆数据"""
        # 处理用户记忆
        if "memories" in user_profile_data:
            for memory in user_profile_data["memories"]:
                self._process_and_store_memory(memory, "user")

        # 处理自衍体记忆
        if "self_memories" in self_profile_data:
            for memory in self_profile_data["self_memories"]:
                self._process_and_store_memory(memory, "self")

    def _process_and_store_memory(self, memory: Dict, source: str) -> str:
        """处理并存储单个记忆"""
        # 生成记忆ID
        memory_id = f"{source}_memory_{hashlib.md5(str(memory).encode()).hexdigest()[:8]}"

        # 准备记忆数据
        processed = {
            "event_summary": memory.get("summary", "未知事件"),
            "raw_content": memory.get("content", ""),
            "tags": self._extract_tags_from_memory(memory),
            "source": source,
            "emotional_intensity": memory.get("emotional_intensity", 0.5),
            "strategic_value": self._assess_strategic_value(memory),
            "linked_tool": self._link_to_cognitive_tools(memory),
            "timestamp": memory.get("timestamp", datetime.now().isoformat())
        }

        # 准备文本用于向量化
        memory_text = f"{processed['event_summary']}。{processed['raw_content']}"

        # 准备元数据
        metadata = {
            'type': 'experience',
            'source': source,
            'summary': processed['event_summary'],
            'tags': processed['tags'],
            'emotional_intensity': processed['emotional_intensity'],
            'strategic_value': processed['strategic_value'],
            'linked_tool': processed['linked_tool'],
            'timestamp': processed['timestamp']
        }

        # 添加到Faiss记忆系统
        _, vector_id = self.memory_system.add_experience_memory(
            {
                'summary': processed['event_summary'],
                'content': processed['raw_content'],
                'source': source,
                'emotional_intensity': processed['emotional_intensity'],
                'tags': processed['tags'],
                'strategic_value': processed['strategic_value'],
                'linked_tool': processed['linked_tool'],
                'timestamp': processed['timestamp']
            },
            self.session_id
        )

        # 更新本地语义记忆地图
        self.semantic_memory_map[memory_id] = {
            **processed,
            'vector_id': vector_id
        }

        # 更新战略标签
        for tag in processed['tags']:
            if tag not in self.strategic_tags:
                self.strategic_tags.append(tag)

        return memory_id

    def _extract_tags_from_memory(self, memory: Dict) -> List[str]:
        """从记忆中提取标签"""
        tags = []
        content = memory.get("content", "").lower()
        summary = memory.get("summary", "").lower()

        # 基于内容的标签
        tag_patterns = [
            ("童年", ["小时候", "童年", "儿时"]),
            ("成就", ["成功", "获奖", "突破", "第一"]),
            ("创伤", ["受伤", "伤害", "痛苦", "失去"]),
            ("学习", ["学习", "读书", "研究", "探索"]),
            ("关系", ["朋友", "家人", "恋人", "同事"]),
            ("挑战", ["困难", "挑战", "压力", "危机"]),
            ("成长", ["成长", "进步", "改变", "成熟"]),
            ("信任", ["信任", "信赖", "相信"]),
            ("背叛", ["背叛", "欺骗", "失望"]),
            ("快乐", ["开心", "快乐", "幸福", "高兴"])
        ]

        full_text = content + " " + summary

        for tag_name, keywords in tag_patterns:
            if any(keyword in full_text for keyword in keywords):
                tags.append(tag_name)

        # 添加情感标签
        emotional_context = memory.get("emotional_context", {})
        if emotional_context.get("valence") == "positive":
            tags.append("积极情感")
        elif emotional_context.get("valence") == "negative":
            tags.append("消极情感")

        if emotional_context.get("intensity", 0) > 0.7:
            tags.append("高情感强度")

        return tags

    def _assess_strategic_value(self, memory: Dict) -> Dict:
        """评估记忆的战略价值"""
        value_score = 0

        # 基于情感强度
        emotional_intensity = memory.get("emotional_intensity", 0.5)
        if emotional_intensity > 0.8:
            value_score += 3
        elif emotional_intensity > 0.6:
            value_score += 2

        # 基于类型
        content = memory.get("content", "").lower()
        if any(word in content for word in ["成功", "成就", "胜利"]):
            value_score += 2
        if any(word in content for word in ["失败", "错误", "教训"]):
            value_score += 1

        # 确定价值级别
        if value_score >= 4:
            level = "核心"
        elif value_score >= 2:
            level = "高"
        elif value_score >= 1:
            level = "中"
        else:
            level = "低"

        return {
            "level": level,
            "score": value_score
        }

    def _link_to_cognitive_tools(self, memory: Dict) -> Optional[str]:
        """链接到认知工具"""
        content = memory.get("content", "").lower()

        tool_patterns = [
            ("情感共情过载", ["受伤", "痛苦", "脆弱", "哭泣"]),
            ("过度保护倾向", ["保护", "关心", "担心", "安全"]),
            ("技术乐观主义", ["成功", "成就", "学习", "探索"])
        ]

        for tool_name, keywords in tool_patterns:
            if any(keyword in content for keyword in keywords):
                return tool_name

        return None

    def find_resonant_memory(self, current_context: Dict) -> Optional[Dict]:
        """
        寻找共鸣记忆（使用Faiss向量搜索）

        Args:
            current_context: 当前情境，包含外部情境和内部状态

        Returns:
            共鸣记忆包或None
        """
        # 构建查询文本
        query_text = self._build_query_from_context(current_context)

        if not query_text:
            return None

        # 在Faiss中搜索相似的经历记忆
        similar_experiences = self.memory_system.find_relevant_experiences(
            {
                'user_input': query_text,
                'emotion': current_context.get('external_context', {}).get('user_emotion', ''),
                'topic': current_context.get('external_context', {}).get('topic_summary', '')
            },
            top_k=3
        )

        if not similar_experiences:
            return None

        # 选择最相似的记忆
        best_match = similar_experiences[0]

        # 构建战术信息包
        tactical_package = self._build_tactical_package(best_match, current_context)

        # 更新访问计数
        if best_match['memory_id'] in self.semantic_memory_map:
            self.semantic_memory_map[best_match['memory_id']]['access_count'] = \
                self.semantic_memory_map[best_match['memory_id']].get('access_count', 0) + 1

        return tactical_package

    def _build_query_from_context(self, context: Dict) -> str:
        """从上下文中构建查询文本"""
        query_parts = []

        # 外部情境
        external = context.get('external_context', {})
        if external:
            query_parts.append(f"用户情绪: {external.get('user_emotion_display', '')}")
            query_parts.append(f"交互类型: {external.get('interaction_type_display', '')}")
            query_parts.append(f"话题: {external.get('topic_summary', '')}")

        # 内部状态
        internal_tags = context.get('internal_state_tags', [])
        if internal_tags:
            query_parts.append(f"自衍体状态: {' '.join(internal_tags[:2])}")

        # 当前目标
        current_goal = context.get('current_goal', '')
        if current_goal:
            query_parts.append(f"目标: {current_goal}")

        return " ".join(query_parts)

    def _build_tactical_package(self, memory_match: Dict, context: Dict) -> Dict:
        """构建战术信息包"""
        metadata = memory_match['metadata']

        package = {
            "triggered_memory": metadata.get('summary', '未知记忆'),
            "memory_id": memory_match['memory_id'],
            "relevance_score": memory_match['similarity'],
            "similarity_distance": memory_match['distance'],
            "source": metadata.get('source', 'unknown'),
            "tags": metadata.get('tags', []),
            "strategic_value": metadata.get('strategic_value', {}),
            "linked_tool": metadata.get('linked_tool'),
            "emotional_intensity": metadata.get('emotional_intensity', 0.5),
            "risk_assessment": self._assess_memory_risk(metadata),
            "recommended_actions": self._generate_recommendations(metadata, context),
            "timestamp": datetime.now().isoformat(),
            "retrieval_method": "faiss_vector_search"
        }

        # 添加风险警告
        risk_level = package["risk_assessment"]["level"]
        if risk_level == "高":
            package["cognitive_alert"] = "高风险记忆，需谨慎使用！"
        elif risk_level == "中":
            package["cognitive_alert"] = "中等风险，建议有策略地使用。"

        return package

    def _assess_memory_risk(self, metadata: Dict) -> Dict:
        """评估记忆使用风险"""
        risk_score = 0

        # 高风险标签
        high_risk_tags = ["创伤", "背叛", "失败", "痛苦"]
        tags = metadata.get('tags', [])
        for tag in tags:
            if tag in high_risk_tags:
                risk_score += 3
            elif "风险" in tag or "危机" in tag:
                risk_score += 2

        # 情感强度影响
        emotional_intensity = metadata.get('emotional_intensity', 0.5)
        if emotional_intensity > 0.8:
            risk_score += 2
        elif emotional_intensity > 0.6:
            risk_score += 1

        # 确定风险级别
        if risk_score >= 5:
            level = "高"
        elif risk_score >= 3:
            level = "中"
        elif risk_score >= 1:
            level = "低"
        else:
            level = "极低"

        return {
            "level": level,
            "score": risk_score,
            "high_risk_factors": [tag for tag in tags if tag in high_risk_tags]
        }

    def _generate_recommendations(self, metadata: Dict, context: Dict) -> List[str]:
        """生成使用建议"""
        recommendations = []
        tags = metadata.get('tags', [])

        # 基于标签的建议
        if "成就" in tags or "成功" in tags:
            recommendations.append("可安全提及以激活积极情绪和TR向量")

        if "创伤" in tags or "痛苦" in tags:
            recommendations.append("高风险区域，仅在高信任度下谨慎触碰")

        if "学习" in tags or "成长" in tags:
            recommendations.append("适合用于激励或共情场景")

        if "信任" in tags or "关系" in tags:
            recommendations.append("适合用于加深CS链接")

        # 基于战略价值的建议
        strategic_value = metadata.get('strategic_value', {}).get('level', '低')
        if strategic_value == "核心":
            recommendations.append("核心战略资产，使用需精确计算")
        elif strategic_value == "高":
            recommendations.append("高价值记忆，可用于关键时刻")

        # 基于链接工具的建议
        linked_tool = metadata.get('linked_tool')
        if linked_tool:
            recommendations.append(f"与认知工具『{linked_tool}』有潜在链接")

        # 基于当前情境的建议
        user_emotion = context.get('external_context', {}).get('user_emotion', 'neutral')
        if user_emotion == "sad" and "创伤" in tags:
            recommendations.append("当前用户情绪低落，避免直接提及创伤记忆")

        return recommendations if recommendations else ["常规记忆，可灵活使用"]

    def add_conversation_to_memory(self, conversation: Dict):
        """将对话添加到记忆系统"""
        self.memory_system.add_conversation_memory(conversation, self.session_id)

        # 也可以从对话中提取关键信息作为经历记忆
        key_insights = self._extract_insights_from_conversation(conversation)
        if key_insights:
            self._process_and_store_memory(key_insights, "conversation")

    def _extract_insights_from_conversation(self, conversation: Dict) -> Optional[Dict]:
        """从对话中提取关键洞察作为记忆"""
        user_input = conversation.get('user_input', '')
        system_response = conversation.get('system_response', '')

        # 简单的启发式：寻找情感强烈的表达或重要信息
        emotional_keywords = ["爱", "恨", "伤心", "开心", "生气", "害怕", "重要", "关键"]

        has_emotional_content = any(keyword in user_input.lower()
                                    for keyword in emotional_keywords)

        if has_emotional_content and len(user_input) > 20:
            return {
                "summary": f"情感对话: {user_input[:50]}...",
                "content": f"用户说: {user_input}\n自衍体回应: {system_response}",
                "emotional_intensity": 0.7,
                "timestamp": conversation.get('timestamp', datetime.now().isoformat())
            }

        return None