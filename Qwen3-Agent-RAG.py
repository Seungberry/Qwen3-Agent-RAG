import os
import re
import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader


# ==================== 配置类 ====================

class TaskType(Enum):
    """任务类型枚举"""
    SIMPLE_QA = "simple_qa"           # 简单问答
    COMPLEX_REASONING = "complex"      # 复杂推理
    MATH_CALCULATION = "math"          # 数学计算
    MULTI_HOP = "multi_hop"            # 多跳查询
    CHITCHAT = "chitchat"              # 闲聊


@dataclass
class AgentConfig:
    """Agent配置"""
    model_name: str = "Qwen/Qwen3-7B-Instruct"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检索参数
    default_top_k: int = 5
    default_threshold: float = 0.7
    max_top_k: int = 10
    min_threshold: float = 0.5
    
    # 置信度阈值
    confidence_threshold: float = 0.7
    max_retry: int = 3
    
    # 记忆配置
    short_term_memory_size: int = 5


# ==================== 工具类定义 ====================

class BaseTool:
    """工具基类"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class SmartRetriever(BaseTool):
    """智能检索工具"""
    
    def __init__(self, vectorstore, config: AgentConfig):
        super().__init__(
            name="smart_retriever",
            description="根据查询内容动态调整检索参数，返回相关文档"
        )
        self.vectorstore = vectorstore
        self.config = config
    
    def execute(self, query: str, top_k: int = None, 
                threshold: float = None, rerank: bool = False) -> Dict[str, Any]:
        """执行智能检索"""
        top_k = top_k or self.config.default_top_k
        threshold = threshold or self.config.default_threshold
        
        # 执行检索
        docs = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        # 过滤低相似度文档
        filtered_docs = [(doc, score) for doc, score in docs if score >= threshold]
        
        # 重排序（如果启用）
        if rerank and len(filtered_docs) > 1:
            filtered_docs = self._rerank(query, filtered_docs)
        
        return {
            "documents": [doc.page_content for doc, _ in filtered_docs],
            "scores": [score for _, score in filtered_docs],
            "count": len(filtered_docs)
        }
    
    def _rerank(self, query: str, docs: List[Tuple]) -> List[Tuple]:
        """简单的重排序实现"""
        # 这里可以使用更复杂的重排序模型
        return sorted(docs, key=lambda x: x[1], reverse=True)


class KnowledgeGraphTool(BaseTool):
    """知识图谱查询工具"""
    
    def __init__(self):
        super().__init__(
            name="knowledge_graph",
            description="处理需要实体关系推理的复杂查询"
        )
        # 简化的知识图谱存储
        self.graph = {}
    
    def execute(self, entities: List[str], relation: str = None, 
                depth: int = 1) -> Dict[str, Any]:
        """执行知识图谱查询"""
        results = []
        for entity in entities:
            if entity in self.graph:
                entity_info = self.graph[entity]
                if relation:
                    if relation in entity_info.get("relations", {}):
                        results.append({
                            "entity": entity,
                            "relation": relation,
                            "targets": entity_info["relations"][relation]
                        })
                else:
                    results.append({
                        "entity": entity,
                        "relations": entity_info.get("relations", {})
                    })
        
        return {
            "results": results,
            "entity_count": len(entities),
            "found_count": len(results)
        }
    
    def add_entity(self, entity: str, relations: Dict):
        """添加实体到知识图谱"""
        self.graph[entity] = {"relations": relations}


class CalculatorTool(BaseTool):
    """计算工具"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算和数值推理"
        )
    
    def execute(self, expression: str, variables: Dict = None) -> Dict[str, Any]:
        """执行计算"""
        try:
            # 安全的计算环境
            safe_dict = {
                'abs': abs, 'max': max, 'min': min,
                'sum': sum, 'len': len, 'round': round,
                'pow': pow, 'sqrt': lambda x: x ** 0.5
            }
            
            if variables:
                safe_dict.update(variables)
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            return {
                "success": True,
                "result": result,
                "expression": expression
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expression": expression
            }


# ==================== Agent核心类 ====================

class TaskClassifier:
    """任务分类器"""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        
        self.classification_prompt = """请分析以下用户查询，判断其任务类型：

用户查询：{query}

可选类型：
1. simple_qa - 简单问答（事实性、单知识点查询）
2. complex - 复杂推理（需要多步骤推理）
3. math - 数学计算（涉及数值计算）
4. multi_hop - 多跳查询（需要跨文档关联）
5. chitchat - 闲聊对话（非知识性问题）

请只输出类型名称，不要输出其他内容。"""
    
    def classify(self, query: str) -> TaskType:
        """分类任务类型"""
        prompt = self.classification_prompt.format(query=query)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("类型名称")[-1].strip().lower()
        
        # 解析类型
        if "math" in response or "计算" in response:
            return TaskType.MATH_CALCULATION
        elif "multi" in response or "多跳" in response:
            return TaskType.MULTI_HOP
        elif "complex" in response or "复杂" in response:
            return TaskType.COMPLEX_REASONING
        elif "chitchat" in response or "闲聊" in response:
            return TaskType.CHITCHAT
        else:
            return TaskType.SIMPLE_QA


class ComplexityEvaluator:
    """复杂度评估器"""
    
    def evaluate(self, query: str, task_type: TaskType) -> Dict[str, Any]:
        """评估查询复杂度"""
        score = 0
        factors = []
        
        # 基于任务类型的基础分数
        type_scores = {
            TaskType.SIMPLE_QA: 1,
            TaskType.CHITCHAT: 1,
            TaskType.MATH_CALCULATION: 2,
            TaskType.COMPLEX_REASONING: 3,
            TaskType.MULTI_HOP: 3
        }
        score += type_scores.get(task_type, 2)
        
        # 查询长度
        if len(query) > 50:
            score += 1
            factors.append("长查询")
        
        # 实体数量
        entities = self._extract_entities(query)
        if len(entities) > 2:
            score += 1
            factors.append(f"多实体({len(entities)}个)")
        
        # 关键词检测
        complex_keywords = ["比较", "对比", "分析", "原因", "影响", "关系"]
        for keyword in complex_keywords:
            if keyword in query:
                score += 1
                factors.append(f"含复杂关键词'{keyword}'")
                break
        
        # 确定复杂度级别
        if score <= 2:
            level = "低"
        elif score <= 4:
            level = "中"
        else:
            level = "高"
        
        return {
            "level": level,
            "score": score,
            "factors": factors,
            "entities": entities
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """简单实体提取"""
        # 使用正则表达式提取可能的实体（大写字母开头的词、引号内的内容等）
        entities = []
        
        # 引号内的内容
        quoted = re.findall(r'["""]([^"""]+)["""]', query)
        entities.extend(quoted)
        
        # 大写字母开头的词（可能是专有名词）
        capitalized = re.findall(r'[A-Z][a-zA-Z]*', query)
        entities.extend(capitalized)
        
        return list(set(entities))


class ConfidenceEvaluator:
    """置信度评估器"""
    
    def evaluate(self, answer: str, retrieved_docs: List[str], 
                 query: str) -> float:
        """评估答案置信度"""
        scores = []
        
        # 1. 不确定性标记检测
        uncertainty_markers = ["可能", "也许", "不确定", "不知道", "无法", "没有相关信息"]
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in answer)
        certainty_score = max(0, 1 - uncertainty_count * 0.2)
        scores.append(certainty_score)
        
        # 2. 答案长度合理性
        if len(answer) < 20:
            scores.append(0.5)
        elif len(answer) > 500:
            scores.append(0.7)
        else:
            scores.append(1.0)
        
        # 3. 检索文档相关性（简化版）
        if retrieved_docs:
            doc_score = min(1.0, len(retrieved_docs) / 3)
            scores.append(doc_score)
        else:
            scores.append(0.0)
        
        # 综合得分
        return sum(scores) / len(scores)


class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.short_term_memory = []  # 短期记忆（当前会话）
        self.long_term_memory = {}   # 长期记忆（用户偏好等）
    
    def add_to_short_term(self, query: str, answer: str, 
                          task_type: TaskType):
        """添加到短期记忆"""
        self.short_term_memory.append({
            "query": query,
            "answer": answer,
            "task_type": task_type.value
        })
        
        # 限制记忆大小
        if len(self.short_term_memory) > self.config.short_term_memory_size:
            self.short_term_memory.pop(0)
    
    def get_context(self) -> str:
        """获取记忆上下文"""
        if not self.short_term_memory:
            return ""
        
        context = "历史对话:\n"
        for item in self.short_term_memory:
            context += f"Q: {item['query']}\nA: {item['answer']}\n\n"
        return context
    
    def clear_short_term(self):
        """清空短期记忆"""
        self.short_term_memory = []


class AgentRAG:
    """Agent+RAG主类"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        
        # 初始化模型
        self._init_models()
        
        # 初始化工具
        self.tools: Dict[str, BaseTool] = {}
        
        # 初始化组件
        self.task_classifier = TaskClassifier(self.tokenizer, self.model)
        self.complexity_evaluator = ComplexityEvaluator()
        self.confidence_evaluator = ConfidenceEvaluator()
        self.memory_manager = MemoryManager(self.config)
        
        # 向量数据库（需要外部设置）
        self.vectorstore = None
    
    def _init_models(self):
        """初始化模型"""
        print("正在加载模型...")
        
        # 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model
        )
        
        print("模型加载完成")
    
    def setup_vectorstore(self, documents: List[str]):
        """设置向量数据库"""
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        texts = []
        for doc in documents:
            texts.extend(text_splitter.split_text(doc))
        
        # 创建向量数据库
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings
        )
        
        # 初始化检索工具
        self.tools["retriever"] = SmartRetriever(self.vectorstore, self.config)
        self.tools["kg"] = KnowledgeGraphTool()
        self.tools["calculator"] = CalculatorTool()
        
        print(f"向量数据库初始化完成，共{len(texts)}个文档块")
    
    def query(self, user_query: str, retry_count: int = 0) -> Dict[str, Any]:
        """处理用户查询"""
        
        # 步骤1: 任务分类
        task_type = self.task_classifier.classify(user_query)
        
        # 步骤2: 复杂度评估
        complexity = self.complexity_evaluator.evaluate(user_query, task_type)
        
        # 步骤3: 确定检索参数
        retrieval_params = self._get_retrieval_params(complexity)
        
        # 步骤4: 执行检索
        retrieved_docs = []
        if self.vectorstore:
            retriever = self.tools["retriever"]
            result = retriever.execute(user_query, **retrieval_params)
            retrieved_docs = result["documents"]
        
        # 步骤5: 工具调用（根据任务类型）
        tool_results = []
        
        if task_type == TaskType.MATH_CALCULATION:
            # 提取数学表达式并计算
            calc_result = self._handle_math_query(user_query, retrieved_docs)
            tool_results.append(calc_result)
        
        elif task_type == TaskType.MULTI_HOP:
            # 多跳查询处理
            hop_result = self._handle_multi_hop(user_query, complexity["entities"])
            tool_results.append(hop_result)
        
        # 步骤6: 生成答案
        answer = self._generate_answer(
            user_query, 
            retrieved_docs, 
            tool_results,
            task_type
        )
        
        # 步骤7: 置信度评估
        confidence = self.confidence_evaluator.evaluate(
            answer, retrieved_docs, user_query
        )
        
        # 步骤8: 失败重试
        if confidence < self.config.confidence_threshold and retry_count < self.config.max_retry:
            print(f"置信度{confidence:.2f}低于阈值，触发重试({retry_count + 1}/{self.config.max_retry})")
            
            # 调整参数后重试
            retrieval_params = self._adjust_retrieval_params(retrieval_params, retry_count)
            return self.query(user_query, retry_count + 1)
        
        # 步骤9: 更新记忆
        self.memory_manager.add_to_short_term(user_query, answer, task_type)
        
        return {
            "query": user_query,
            "answer": answer,
            "task_type": task_type.value,
            "complexity": complexity,
            "confidence": confidence,
            "retrieved_docs_count": len(retrieved_docs),
            "tool_results": tool_results,
            "retry_count": retry_count
        }
    
    def _get_retrieval_params(self, complexity: Dict) -> Dict:
        """根据复杂度获取检索参数"""
        level = complexity["level"]
        
        params_map = {
            "低": {"top_k": 3, "threshold": 0.75, "rerank": False},
            "中": {"top_k": 5, "threshold": 0.65, "rerank": True},
            "高": {"top_k": 8, "threshold": 0.55, "rerank": True}
        }
        
        return params_map.get(level, params_map["中"])
    
    def _adjust_retrieval_params(self, params: Dict, retry_count: int) -> Dict:
        """调整检索参数用于重试"""
        new_params = params.copy()
        
        if retry_count == 1:
            # 第一次重试：降低阈值，增加top_k
            new_params["threshold"] = max(0.5, params["threshold"] - 0.1)
            new_params["top_k"] = min(10, params["top_k"] + 2)
        elif retry_count == 2:
            # 第二次重试：启用混合检索（简化实现）
            new_params["threshold"] = 0.5
            new_params["top_k"] = 10
            new_params["rerank"] = True
        
        return new_params
    
    def _handle_math_query(self, query: str, docs: List[str]) -> Dict:
        """处理数学查询"""
        # 尝试从文档中提取数字
        numbers = re.findall(r'\d+\.?\d*', " ".join(docs))
        
        # 尝试提取数学表达式
        expression = self._extract_expression(query)
        
        if expression:
            calculator = self.tools["calculator"]
            result = calculator.execute(expression)
            return {
                "tool": "calculator",
                "result": result
            }
        
        return {"tool": "calculator", "result": None}
    
    def _extract_expression(self, query: str) -> Optional[str]:
        """从查询中提取数学表达式"""
        # 简单的表达式提取
        patterns = [
            r'(\d+)\s*([+\-*/])\s*(\d+)',
            r'增长[率]*[是\s]*(\d+)%',
            r'([\d\.]+)\s*倍'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(0)
        
        return None
    
    def _handle_multi_hop(self, query: str, entities: List[str]) -> Dict:
        """处理多跳查询"""
        kg = self.tools["kg"]
        result = kg.execute(entities=entities, depth=2)
        
        return {
            "tool": "knowledge_graph",
            "result": result
        }
    
    def _generate_answer(self, query: str, docs: List[str], 
                         tool_results: List[Dict], task_type: TaskType) -> str:
        """生成答案"""
        
        # 构建提示词
        memory_context = self.memory_manager.get_context()
        
        prompt = f"""你是一个智能助手，请根据提供的上下文回答问题。

{memory_context}

参考文档：
{chr(10).join([f"{i+1}. {doc[:200]}..." for i, doc in enumerate(docs[:3])])}

工具调用结果：
{json.dumps(tool_results, ensure_ascii=False, indent=2) if tool_results else "无"}

用户问题：{query}

请给出准确、简洁的回答："""
        
        # 生成答案
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的部分
        answer = answer.split("请给出准确、简洁的回答：")[-1].strip()
        
        return answer
    
    def clear_memory(self):
        """清空记忆"""
        self.memory_manager.clear_short_term()


# ==================== 测试与评估 ====================

def run_tests():
    """运行测试"""
    # 初始化Agent
    config = AgentConfig()
    agent = AgentRAG(config)
    
    # 设置测试文档
    test_docs = [
        "阿里巴巴2023年营收为8686.87亿元人民币，同比增长8%。",
        "阿里巴巴2022年营收为8530.62亿元人民币。",
        "腾讯2023年营收为6090亿元人民币，同比增长10%。",
        "腾讯2022年营收为5537亿元人民币。",
        "百度2023年营收为1345.98亿元人民币，同比增长9%。",
        "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。",
        "机器学习是人工智能的一个子领域，使用统计技术让计算机从数据中学习。",
        "深度学习是机器学习的一种方法，使用多层神经网络进行学习。"
    ]
    
    agent.setup_vectorstore(test_docs)
    
    # 测试用例
    test_cases = [
        {
            "query": "阿里巴巴2023年的营收是多少？",
            "expected_type": "simple_qa"
        },
        {
            "query": "阿里巴巴2023年比2022年营收增长了多少百分比？",
            "expected_type": "math"
        },
        {
            "query": "比较阿里巴巴和腾讯2023年的营收情况",
            "expected_type": "complex"
        },
        {
            "query": "什么是深度学习？",
            "expected_type": "simple_qa"
        }
    ]
    
    print("\n" + "="*50)
    print("开始测试")
    print("="*50 + "\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test['query']}")
        print(f"期望类型: {test['expected_type']}")
        print("-" * 50)
        
        result = agent.query(test['query'])
        
        print(f"实际类型: {result['task_type']}")
        print(f"复杂度: {result['complexity']['level']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"重试次数: {result['retry_count']}")
        print(f"检索文档数: {result['retrieved_docs_count']}")
        print(f"\n回答:\n{result['answer']}")
        print("\n" + "="*50)


if __name__ == "__main__":
    run_tests()