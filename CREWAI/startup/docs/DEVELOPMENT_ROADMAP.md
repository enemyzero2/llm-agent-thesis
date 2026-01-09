# CrewAI 混合多智能体 AI 助手系统 - 开发路线图

## 项目概述

本项目旨在构建一个基于 CrewAI 框架的混合多智能体 AI 助手系统，从基础的 Agent 交互逐步演进到集成 RCI（Recursive Criticism and Improvement）自我反思、RAG（Retrieval-Augmented Generation）知识增强以及 MCP（Model Context Protocol）的完整系统。

## 技术栈

- **核心框架**: CrewAI 1.7.2+
- **语言**: Python 3.10+
- **LLM**: OpenAI GPT-4 / Claude / 本地模型
- **向量数据库**: ChromaDB / FAISS
- **知识管理**: LangChain / LlamaIndex
- **MCP 集成**: Anthropic MCP SDK

## 开发阶段总览

```
阶段一 ──► 阶段二 ──► 阶段三 ──► 阶段四 ──► 阶段五 ──► 阶段六
基础Agent   多Agent    RCI反思    RAG增强    MCP集成    混合决策
  交互       协作       模块       系统       协议       系统
```

---

## 阶段一：基础 Agent 交互 (Foundation)

### 目标
- 理解 CrewAI Agent 的基本概念
- 掌握 Agent 配置与任务定义
- 实现单 Agent 的基本对话能力

### 核心概念

```
┌─────────────────────────────────────────────────────────┐
│                      Agent 架构                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │  Role   │    │  Goal   │    │Backstory│             │
│  │  角色   │    │  目标   │    │  背景   │             │
│  └────┬────┘    └────┬────┘    └────┬────┘             │
│       │              │              │                   │
│       └──────────────┼──────────────┘                   │
│                      ▼                                  │
│              ┌──────────────┐                           │
│              │    Agent     │                           │
│              │   智能体     │                           │
│              └──────┬───────┘                           │
│                     │                                   │
│         ┌───────────┼───────────┐                       │
│         ▼           ▼           ▼                       │
│    ┌────────┐  ┌────────┐  ┌────────┐                  │
│    │ Tools  │  │  LLM   │  │ Memory │                  │
│    │  工具  │  │  模型  │  │  记忆  │                  │
│    └────────┘  └────────┘  └────────┘                  │
└─────────────────────────────────────────────────────────┘
```

### 学习要点

1. **Agent 三要素**
   - Role（角色）: 定义 Agent 的身份
   - Goal（目标）: Agent 要达成的目标
   - Backstory（背景）: Agent 的专业背景和经验

2. **Task 定义**
   - Description: 任务描述
   - Expected Output: 期望输出格式
   - Agent: 执行任务的 Agent

### 目录结构

```
phase1_basic_agent/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── basic_agent.py      # 基础Agent定义
│   ├── config/
│   │   ├── agents.yaml         # Agent配置
│   │   └── tasks.yaml          # 任务配置
│   └── main.py                 # 入口文件
├── tests/
│   └── test_basic_agent.py
└── README.md
```

### 代码示例

#### 1. 基础 Agent 配置 (agents.yaml)

```yaml
# =============================================================================
# 阶段一：基础Agent配置
# 文件: config/agents.yaml
# 说明: 定义AI助手的基本角色、目标和背景
# =============================================================================

assistant:
  role: >
    智能AI助手
  goal: >
    帮助用户解答问题、提供建议，并以友好专业的方式完成用户交代的任务
  backstory: >
    你是一位经验丰富的AI助手，具备广泛的知识储备和出色的沟通能力。
    你善于理解用户需求，能够提供准确、有帮助的回答。
    你的回答总是清晰、有条理，并且会考虑用户的具体情况。
```

#### 2. 任务配置 (tasks.yaml)

```yaml
# =============================================================================
# 阶段一：基础任务配置
# 文件: config/tasks.yaml
# 说明: 定义Agent需要执行的任务
# =============================================================================

chat_task:
  description: >
    用户问题: {user_input}

    请仔细分析用户的问题，提供准确、有帮助的回答。
    如果问题不清楚，请礼貌地请求澄清。
  expected_output: >
    一个清晰、有条理的回答，直接解决用户的问题或需求。
  agent: assistant
```

#### 3. 基础 Agent 实现 (basic_agent.py)

```python
"""
=============================================================================
阶段一：基础Agent实现
文件: src/agents/basic_agent.py
说明: 实现最基础的单Agent对话系统
=============================================================================
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List


@CrewBase
class BasicAssistantCrew:
    """
    基础AI助手Crew

    这是最简单的单Agent实现，用于理解CrewAI的基本工作原理。

    核心组件:
    - Agent: 定义智能体的角色、目标和背景
    - Task: 定义Agent需要完成的任务
    - Crew: 组织Agent和Task的执行
    """

    # =========================================================================
    # Agent 定义
    # =========================================================================

    @agent
    def assistant(self) -> Agent:
        """
        创建基础AI助手Agent

        Agent的三个核心属性:
        1. role (角色): 定义Agent的身份，影响其行为方式
        2. goal (目标): Agent要达成的目标，指导其决策
        3. backstory (背景): 提供上下文，让Agent更好地理解自己的定位

        Returns:
            Agent: 配置好的AI助手Agent实例
        """
        return Agent(
            config=self.agents_config['assistant'],  # 从YAML加载配置
            verbose=True,  # 开启详细日志，便于调试
            # memory=True,  # 阶段一暂不启用记忆功能
            # tools=[],     # 阶段一暂不添加工具
        )

    # =========================================================================
    # Task 定义
    # =========================================================================

    @task
    def chat_task(self) -> Task:
        """
        创建对话任务

        Task的核心属性:
        1. description: 任务描述，告诉Agent需要做什么
        2. expected_output: 期望的输出格式
        3. agent: 执行此任务的Agent

        Returns:
            Task: 配置好的对话任务实例
        """
        return Task(
            config=self.tasks_config['chat_task'],
        )

    # =========================================================================
    # Crew 定义
    # =========================================================================

    @crew
    def crew(self) -> Crew:
        """
        创建并配置Crew

        Crew是Agent和Task的组织者，负责:
        1. 管理Agent列表
        2. 管理Task列表
        3. 定义执行流程 (sequential/hierarchical)

        Returns:
            Crew: 配置好的Crew实例
        """
        return Crew(
            agents=self.agents,    # 自动收集所有@agent装饰的方法
            tasks=self.tasks,      # 自动收集所有@task装饰的方法
            process=Process.sequential,  # 顺序执行任务
            verbose=True,
        )


# =============================================================================
# 运行入口
# =============================================================================

def run_basic_assistant(user_input: str) -> str:
    """
    运行基础AI助手

    Args:
        user_input: 用户输入的问题或请求

    Returns:
        str: AI助手的回答
    """
    inputs = {
        'user_input': user_input
    }

    result = BasicAssistantCrew().crew().kickoff(inputs=inputs)
    return result.raw


if __name__ == "__main__":
    # 简单测试
    response = run_basic_assistant("你好，请介绍一下你自己")
    print(response)
```

---

## 阶段二：多 Agent 协作 (Multi-Agent Collaboration)

### 目标
- 实现多个 Agent 之间的协作
- 理解任务委派和信息传递
- 掌握 Sequential 和 Hierarchical 两种执行模式

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        多Agent协作架构                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Sequential Process (顺序执行)                  │   │
│  │                                                                   │   │
│  │   Task1 ──► Agent1 ──► Output1 ──► Task2 ──► Agent2 ──► Output2  │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  Hierarchical Process (层级执行)                  │   │
│  │                                                                   │   │
│  │                      ┌──────────────┐                            │   │
│  │                      │   Manager    │                            │   │
│  │                      │   管理者     │                            │   │
│  │                      └──────┬───────┘                            │   │
│  │                             │                                     │   │
│  │              ┌──────────────┼──────────────┐                     │   │
│  │              ▼              ▼              ▼                     │   │
│  │        ┌─────────┐   ┌─────────┐   ┌─────────┐                  │   │
│  │        │ Agent1  │   │ Agent2  │   │ Agent3  │                  │   │
│  │        │ 研究员  │   │ 分析师  │   │ 执行者  │                  │   │
│  │        └─────────┘   └─────────┘   └─────────┘                  │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
phase2_multi_agent/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── researcher.py       # 研究员Agent
│   │   ├── analyst.py          # 分析师Agent
│   │   ├── executor.py         # 执行者Agent
│   │   └── manager.py          # 管理者Agent (层级模式)
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   ├── crews/
│   │   ├── __init__.py
│   │   ├── sequential_crew.py  # 顺序执行Crew
│   │   └── hierarchical_crew.py # 层级执行Crew
│   └── main.py
├── tests/
│   ├── test_sequential.py
│   └── test_hierarchical.py
└── README.md
```

### 代码示例

#### 1. 多Agent配置 (agents.yaml)

```yaml
# =============================================================================
# 阶段二：多Agent配置
# 文件: config/agents.yaml
# 说明: 定义多个协作Agent，实现复杂任务分解
# =============================================================================

# -----------------------------------------------------------------------------
# 研究员Agent - 负责信息收集和初步分析
# -----------------------------------------------------------------------------
researcher:
  role: >
    高级信息研究员
  goal: >
    深入研究用户提出的问题，收集全面、准确的相关信息
  backstory: >
    你是一位资深的信息研究专家，拥有多年的研究经验。
    你擅长从海量信息中提取关键内容，能够快速定位问题的核心。
    你的研究报告总是全面、客观、有据可查。

# -----------------------------------------------------------------------------
# 分析师Agent - 负责深度分析和方案制定
# -----------------------------------------------------------------------------
analyst:
  role: >
    策略分析师
  goal: >
    基于研究结果进行深度分析，提出可行的解决方案
  backstory: >
    你是一位经验丰富的策略分析师，擅长将复杂问题分解为可执行的步骤。
    你能够从多个角度分析问题，权衡利弊，提出最优解决方案。
    你的分析报告逻辑清晰，建议具有很强的可操作性。

# -----------------------------------------------------------------------------
# 执行者Agent - 负责方案执行和结果输出
# -----------------------------------------------------------------------------
executor:
  role: >
    任务执行专家
  goal: >
    将分析结果转化为具体的执行方案，并生成最终输出
  backstory: >
    你是一位高效的执行专家，擅长将策略转化为行动。
    你注重细节，能够确保每个步骤都得到正确执行。
    你的输出总是清晰、完整、易于理解。

# -----------------------------------------------------------------------------
# 管理者Agent - 层级模式下的任务协调者
# -----------------------------------------------------------------------------
manager:
  role: >
    项目经理
  goal: >
    协调团队成员，确保任务高效完成
  backstory: >
    你是一位出色的项目经理，擅长任务分配和团队协调。
    你能够根据每个成员的特长分配任务，确保整体效率最大化。
    你善于把控进度，及时发现和解决问题。
```

#### 2. 多任务配置 (tasks.yaml)

```yaml
# =============================================================================
# 阶段二：多任务配置
# 文件: config/tasks.yaml
# 说明: 定义多个相互关联的任务，实现任务链
# =============================================================================

# -----------------------------------------------------------------------------
# 研究任务 - 第一步：信息收集
# -----------------------------------------------------------------------------
research_task:
  description: >
    用户需求: {user_request}

    请对上述需求进行深入研究:
    1. 理解用户的核心需求
    2. 收集相关的背景信息
    3. 识别关键问题和挑战
    4. 整理初步的研究发现
  expected_output: >
    一份结构化的研究报告，包含:
    - 需求理解摘要
    - 关键发现列表
    - 潜在挑战识别
    - 初步建议方向
  agent: researcher

# -----------------------------------------------------------------------------
# 分析任务 - 第二步：深度分析
# -----------------------------------------------------------------------------
analysis_task:
  description: >
    基于研究员提供的研究报告，进行深度分析:
    1. 评估各个发现的重要性
    2. 分析可能的解决方案
    3. 权衡各方案的优缺点
    4. 制定推荐的行动计划
  expected_output: >
    一份详细的分析报告，包含:
    - 问题分析总结
    - 解决方案对比
    - 推荐方案及理由
    - 具体行动步骤
  agent: analyst
  context:
    - research_task  # 依赖研究任务的输出

# -----------------------------------------------------------------------------
# 执行任务 - 第三步：生成最终输出
# -----------------------------------------------------------------------------
execution_task:
  description: >
    基于分析师的分析报告，生成最终的用户回复:
    1. 整合所有分析结果
    2. 以用户友好的方式呈现
    3. 提供清晰的行动指南
    4. 确保回复完整且易于理解
  expected_output: >
    一份面向用户的完整回复，包含:
    - 问题解答
    - 具体建议
    - 行动步骤
    - 注意事项
  agent: executor
  context:
    - analysis_task  # 依赖分析任务的输出
```

#### 3. 顺序执行Crew实现 (sequential_crew.py)

```python
"""
=============================================================================
阶段二：顺序执行多Agent Crew
文件: src/crews/sequential_crew.py
说明: 实现多Agent顺序协作，任务按定义顺序依次执行
=============================================================================
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List


@CrewBase
class SequentialAssistantCrew:
    """
    顺序执行的多Agent助手Crew

    执行流程:
    1. Researcher 收集信息
    2. Analyst 分析信息
    3. Executor 生成输出

    每个Agent的输出会作为下一个Agent的输入上下文
    """

    # =========================================================================
    # Agent 定义
    # =========================================================================

    @agent
    def researcher(self) -> Agent:
        """
        研究员Agent

        职责: 信息收集和初步整理
        特点: 广泛搜索、客观记录
        """
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            allow_delegation=False,  # 不允许委派任务给其他Agent
        )

    @agent
    def analyst(self) -> Agent:
        """
        分析师Agent

        职责: 深度分析和方案制定
        特点: 逻辑严密、方案可行
        """
        return Agent(
            config=self.agents_config['analyst'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def executor(self) -> Agent:
        """
        执行者Agent

        职责: 整合输出和用户交付
        特点: 清晰表达、用户友好
        """
        return Agent(
            config=self.agents_config['executor'],
            verbose=True,
            allow_delegation=False,
        )

    # =========================================================================
    # Task 定义
    # =========================================================================

    @task
    def research_task(self) -> Task:
        """研究任务 - 第一步"""
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def analysis_task(self) -> Task:
        """分析任务 - 第二步"""
        return Task(
            config=self.tasks_config['analysis_task'],
        )

    @task
    def execution_task(self) -> Task:
        """执行任务 - 第三步"""
        return Task(
            config=self.tasks_config['execution_task'],
        )

    # =========================================================================
    # Crew 定义
    # =========================================================================

    @crew
    def crew(self) -> Crew:
        """
        创建顺序执行的Crew

        Process.sequential 特点:
        - 任务按定义顺序执行
        - 前一个任务的输出自动传递给后续任务
        - 适合有明确依赖关系的任务链
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,  # 顺序执行
            verbose=True,
        )


# =============================================================================
# 运行入口
# =============================================================================

def run_sequential_assistant(user_request: str) -> str:
    """
    运行顺序执行的多Agent助手

    Args:
        user_request: 用户的请求

    Returns:
        str: 最终的处理结果
    """
    inputs = {
        'user_request': user_request
    }

    result = SequentialAssistantCrew().crew().kickoff(inputs=inputs)
    return result.raw
```

#### 4. 层级执行Crew实现 (hierarchical_crew.py)

```python
"""
=============================================================================
阶段二：层级执行多Agent Crew
文件: src/crews/hierarchical_crew.py
说明: 实现层级管理模式，由Manager Agent协调其他Agent
=============================================================================
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List


@CrewBase
class HierarchicalAssistantCrew:
    """
    层级执行的多Agent助手Crew

    执行流程:
    1. Manager 接收任务，分析需求
    2. Manager 将任务分配给合适的Agent
    3. Agent 执行任务并汇报结果
    4. Manager 整合结果，决定下一步

    特点:
    - 更灵活的任务分配
    - 动态决策执行顺序
    - 适合复杂、不确定的任务
    """

    # =========================================================================
    # Agent 定义
    # =========================================================================

    @agent
    def manager(self) -> Agent:
        """
        管理者Agent

        职责: 任务分配和协调
        特点: 全局视角、动态调度

        注意: 在层级模式下，Manager会自动被创建，
        但我们也可以自定义Manager的行为
        """
        return Agent(
            config=self.agents_config['manager'],
            verbose=True,
            allow_delegation=True,  # 允许委派任务
        )

    @agent
    def researcher(self) -> Agent:
        """研究员Agent"""
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def analyst(self) -> Agent:
        """分析师Agent"""
        return Agent(
            config=self.agents_config['analyst'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def executor(self) -> Agent:
        """执行者Agent"""
        return Agent(
            config=self.agents_config['executor'],
            verbose=True,
            allow_delegation=False,
        )

    # =========================================================================
    # Task 定义 (层级模式下任务定义更灵活)
    # =========================================================================

    @task
    def main_task(self) -> Task:
        """
        主任务

        在层级模式下，我们定义一个主任务，
        Manager会根据需要将其分解并分配给其他Agent
        """
        return Task(
            config=self.tasks_config['main_task'],
        )

    # =========================================================================
    # Crew 定义
    # =========================================================================

    @crew
    def crew(self) -> Crew:
        """
        创建层级执行的Crew

        Process.hierarchical 特点:
        - Manager Agent 负责任务分配
        - 动态决定执行顺序
        - 可以根据中间结果调整策略
        - 适合复杂、需要灵活处理的任务
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,  # 层级执行
            manager_agent=self.manager(),  # 指定管理者
            verbose=True,
        )


# =============================================================================
# 运行入口
# =============================================================================

def run_hierarchical_assistant(user_request: str) -> str:
    """
    运行层级执行的多Agent助手

    Args:
        user_request: 用户的请求

    Returns:
        str: 最终的处理结果
    """
    inputs = {
        'user_request': user_request
    }

    result = HierarchicalAssistantCrew().crew().kickoff(inputs=inputs)
    return result.raw
```

---

## 阶段三：RCI 自我反思模块 (Recursive Criticism and Improvement)

### 目标
- 实现 Agent 的自我评估和改进能力
- 构建批评-改进循环机制
- 提高输出质量和可靠性

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RCI 自我反思架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     RCI 循环流程                                 │  │
│   │                                                                  │  │
│   │    ┌──────────┐     ┌──────────┐     ┌──────────┐              │  │
│   │    │  生成    │────►│  批评    │────►│  改进    │              │  │
│   │    │ Generate │     │ Criticize│     │ Improve  │              │  │
│   │    └──────────┘     └────┬─────┘     └────┬─────┘              │  │
│   │         ▲                │                │                     │  │
│   │         │                ▼                │                     │  │
│   │         │         ┌──────────┐            │                     │  │
│   │         │         │ 质量评估 │            │                     │  │
│   │         │         │ Quality  │            │                     │  │
│   │         │         │  Check   │            │                     │  │
│   │         │         └────┬─────┘            │                     │  │
│   │         │              │                  │                     │  │
│   │         │    ┌─────────┴─────────┐       │                     │  │
│   │         │    ▼                   ▼       ▼                     │  │
│   │         │ [不通过]            [通过]                            │  │
│   │         │    │                   │                              │  │
│   │         └────┘                   ▼                              │  │
│   │                           ┌──────────┐                          │  │
│   │                           │ 最终输出 │                          │  │
│   │                           │  Output  │                          │  │
│   │                           └──────────┘                          │  │
│   │                                                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
phase3_rci_reflection/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── generator.py        # 生成Agent
│   │   ├── critic.py           # 批评Agent
│   │   └── improver.py         # 改进Agent
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   ├── rci/
│   │   ├── __init__.py
│   │   ├── rci_loop.py         # RCI循环实现
│   │   └── quality_checker.py  # 质量检查器
│   └── main.py
├── tests/
│   └── test_rci.py
└── README.md
```

### 代码示例

#### 1. RCI Agent配置 (agents.yaml)

```yaml
# =============================================================================
# 阶段三：RCI自我反思Agent配置
# 文件: config/agents.yaml
# 说明: 定义生成、批评、改进三个核心Agent
# =============================================================================

# -----------------------------------------------------------------------------
# 生成Agent - 负责初始内容生成
# -----------------------------------------------------------------------------
generator:
  role: >
    内容生成专家
  goal: >
    根据用户需求生成高质量的初始内容
  backstory: >
    你是一位创意丰富的内容生成专家，擅长快速理解需求并产出内容。
    你的输出总是结构清晰、内容丰富，为后续优化提供良好基础。

# -----------------------------------------------------------------------------
# 批评Agent - 负责质量评估和问题识别
# -----------------------------------------------------------------------------
critic:
  role: >
    质量评审专家
  goal: >
    客观评估内容质量，识别问题和改进空间
  backstory: >
    你是一位严谨的质量评审专家，拥有敏锐的洞察力。
    你能够从多个维度评估内容：准确性、完整性、逻辑性、可读性。
    你的批评总是建设性的，指出问题的同时提供改进方向。

# -----------------------------------------------------------------------------
# 改进Agent - 负责根据批评意见优化内容
# -----------------------------------------------------------------------------
improver:
  role: >
    内容优化专家
  goal: >
    根据批评意见改进内容，提升整体质量
  backstory: >
    你是一位经验丰富的内容优化专家，擅长根据反馈改进内容。
    你能够准确理解批评意见，并将其转化为具体的改进行动。
    你的优化总是有针对性的，确保每次迭代都有明显提升。
```

#### 2. 质量检查器 (quality_checker.py)

```python
"""
=============================================================================
阶段三：质量检查器
文件: src/rci/quality_checker.py
说明: 实现内容质量评估，决定是否需要继续迭代
=============================================================================
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class QualityLevel(Enum):
    """质量等级枚举"""
    EXCELLENT = "excellent"    # 优秀，无需改进
    GOOD = "good"              # 良好，可选改进
    ACCEPTABLE = "acceptable"  # 可接受，建议改进
    POOR = "poor"              # 较差，必须改进


class QualityDimension(BaseModel):
    """
    质量维度评估模型

    用于评估内容在某一维度上的表现
    """
    name: str = Field(description="维度名称")
    score: float = Field(ge=0, le=10, description="得分(0-10)")
    feedback: str = Field(description="具体反馈")
    suggestions: List[str] = Field(default=[], description="改进建议")


class QualityReport(BaseModel):
    """
    质量评估报告模型

    包含多维度评估结果和整体判断
    """
    dimensions: List[QualityDimension] = Field(description="各维度评估")
    overall_score: float = Field(ge=0, le=10, description="综合得分")
    level: QualityLevel = Field(description="质量等级")
    pass_threshold: bool = Field(description="是否通过阈值")
    summary: str = Field(description="评估摘要")


class QualityChecker:
    """
    质量检查器

    负责评估内容质量，决定RCI循环是否继续
    """

    def __init__(
        self,
        threshold: float = 7.0,
        max_iterations: int = 3,
        dimensions: Optional[List[str]] = None
    ):
        """
        初始化质量检查器

        Args:
            threshold: 通过阈值，默认7.0分
            max_iterations: 最大迭代次数
            dimensions: 评估维度列表
        """
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.dimensions = dimensions or [
            "准确性",    # 内容是否准确
            "完整性",    # 内容是否完整
            "逻辑性",    # 逻辑是否清晰
            "可读性",    # 是否易于理解
        ]
        self.iteration_count = 0

    def should_continue(self, report: QualityReport) -> bool:
        """
        判断是否需要继续迭代

        Args:
            report: 质量评估报告

        Returns:
            bool: True表示需要继续，False表示可以结束
        """
        self.iteration_count += 1

        # 达到最大迭代次数，强制结束
        if self.iteration_count >= self.max_iterations:
            return False

        # 通过阈值，可以结束
        if report.pass_threshold:
            return False

        # 未通过阈值，继续迭代
        return True

    def reset(self):
        """重置迭代计数器"""
        self.iteration_count = 0
```

#### 3. RCI循环实现 (rci_loop.py)

```python
"""
=============================================================================
阶段三：RCI循环实现
文件: src/rci/rci_loop.py
说明: 实现递归批评与改进的核心循环逻辑
=============================================================================
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List, Optional
from .quality_checker import QualityChecker, QualityReport


@CrewBase
class RCICrew:
    """
    RCI (Recursive Criticism and Improvement) Crew

    实现生成-批评-改进的迭代循环，持续提升输出质量
    """

    def __init__(self, max_iterations: int = 3, threshold: float = 7.0):
        """
        初始化RCI Crew

        Args:
            max_iterations: 最大迭代次数
            threshold: 质量通过阈值
        """
        self.quality_checker = QualityChecker(
            threshold=threshold,
            max_iterations=max_iterations
        )

    # =========================================================================
    # Agent 定义
    # =========================================================================

    @agent
    def generator(self) -> Agent:
        """生成Agent - 负责内容生成"""
        return Agent(
            config=self.agents_config['generator'],
            verbose=True,
        )

    @agent
    def critic(self) -> Agent:
        """批评Agent - 负责质量评估"""
        return Agent(
            config=self.agents_config['critic'],
            verbose=True,
        )

    @agent
    def improver(self) -> Agent:
        """改进Agent - 负责内容优化"""
        return Agent(
            config=self.agents_config['improver'],
            verbose=True,
        )

    # =========================================================================
    # Task 定义
    # =========================================================================

    @task
    def generate_task(self) -> Task:
        """生成任务 - 创建初始内容"""
        return Task(
            config=self.tasks_config['generate_task'],
        )

    @task
    def criticize_task(self) -> Task:
        """批评任务 - 评估内容质量"""
        return Task(
            config=self.tasks_config['criticize_task'],
            output_pydantic=QualityReport,  # 结构化输出
        )

    @task
    def improve_task(self) -> Task:
        """改进任务 - 优化内容"""
        return Task(
            config=self.tasks_config['improve_task'],
        )

    # =========================================================================
    # Crew 定义
    # =========================================================================

    @crew
    def crew(self) -> Crew:
        """创建RCI Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    # =========================================================================
    # RCI 循环执行
    # =========================================================================

    def run_rci_loop(self, user_request: str) -> dict:
        """
        执行RCI循环

        Args:
            user_request: 用户请求

        Returns:
            dict: 包含最终输出和迭代历史
        """
        self.quality_checker.reset()
        iteration_history = []
        current_content = None

        # 第一次生成
        inputs = {'user_request': user_request}
        result = self.crew().kickoff(inputs=inputs)
        current_content = result.raw

        iteration_history.append({
            'iteration': 1,
            'content': current_content,
            'type': 'initial_generation'
        })

        # RCI循环
        while True:
            # 批评阶段
            critic_inputs = {
                'content_to_review': current_content,
                'original_request': user_request
            }
            critic_result = self._run_critic(critic_inputs)

            # 检查是否需要继续
            if not self.quality_checker.should_continue(critic_result):
                break

            # 改进阶段
            improve_inputs = {
                'content_to_improve': current_content,
                'criticism': critic_result.summary,
                'suggestions': critic_result.dimensions
            }
            improved_content = self._run_improver(improve_inputs)
            current_content = improved_content

            iteration_history.append({
                'iteration': self.quality_checker.iteration_count,
                'content': current_content,
                'quality_report': critic_result.dict(),
                'type': 'improvement'
            })

        return {
            'final_output': current_content,
            'iterations': len(iteration_history),
            'history': iteration_history
        }


# =============================================================================
# 运行入口
# =============================================================================

def run_rci_assistant(user_request: str, max_iterations: int = 3) -> str:
    """
    运行RCI增强的AI助手

    Args:
        user_request: 用户请求
        max_iterations: 最大迭代次数

    Returns:
        str: 经过RCI优化的最终输出
    """
    rci_crew = RCICrew(max_iterations=max_iterations)
    result = rci_crew.run_rci_loop(user_request)
    return result['final_output']
```

---

## 阶段四：RAG 知识增强系统 (Retrieval-Augmented Generation)

### 目标
- 集成向量数据库实现知识存储
- 实现语义检索增强生成
- 构建知识管理和更新机制

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAG 知识增强架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     知识处理流程                                 │  │
│   │                                                                  │  │
│   │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │  │
│   │  │ 文档加载 │───►│ 文本分块 │───►│ 向量化  │───►│ 存储索引 │  │  │
│   │  │  Loader  │    │ Chunking │    │Embedding │    │  Store   │  │  │
│   │  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │  │
│   │                                                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     检索增强流程                                 │  │
│   │                                                                  │  │
│   │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │  │
│   │  │ 用户查询 │───►│ 语义检索 │───►│ 上下文  │───►│ LLM生成  │  │  │
│   │  │  Query   │    │ Retrieve │    │ Context  │    │ Generate │  │  │
│   │  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │  │
│   │                        │                                        │  │
│   │                        ▼                                        │  │
│   │                 ┌──────────────┐                                │  │
│   │                 │  向量数据库  │                                │  │
│   │                 │ ChromaDB/    │                                │  │
│   │                 │   FAISS      │                                │  │
│   │                 └──────────────┘                                │  │
│   │                                                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
phase4_rag_system/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── rag_agent.py        # RAG增强Agent
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py  # 文档加载器
│   │   ├── text_splitter.py    # 文本分块器
│   │   ├── embeddings.py       # 向量化模块
│   │   ├── vector_store.py     # 向量存储
│   │   └── retriever.py        # 检索器
│   ├── tools/
│   │   ├── __init__.py
│   │   └── rag_tool.py         # RAG工具
│   └── main.py
├── knowledge/                   # 知识库目录
│   ├── documents/
│   └── index/
├── tests/
│   └── test_rag.py
└── README.md
```

### 代码示例

#### 1. 向量存储实现 (vector_store.py)

```python
"""
=============================================================================
阶段四：向量存储模块
文件: src/rag/vector_store.py
说明: 实现基于ChromaDB的向量存储和检索
=============================================================================
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from pathlib import Path


class VectorStore:
    """
    向量存储类

    使用ChromaDB实现文档的向量化存储和语义检索
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: Optional[str] = None
    ):
        """
        初始化向量存储

        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录，None则使用内存存储
        """
        # 配置ChromaDB客户端
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
        else:
            self.client = chromadb.Client()

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "AI助手知识库"}
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        添加文档到向量存储

        Args:
            documents: 文档文本列表
            metadatas: 元数据列表
            ids: 文档ID列表
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(
        self,
        query_text: str,
        n_results: int = 5
    ) -> Dict:
        """
        语义检索

        Args:
            query_text: 查询文本
            n_results: 返回结果数量

        Returns:
            Dict: 检索结果
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
```

#### 2. RAG检索工具 (rag_tool.py)

```python
"""
=============================================================================
阶段四：RAG检索工具
文件: src/tools/rag_tool.py
说明: 为Agent提供知识库检索能力
=============================================================================
"""

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from ..rag.vector_store import VectorStore


class RAGToolInput(BaseModel):
    """RAG工具输入模型"""
    query: str = Field(description="检索查询文本")
    top_k: int = Field(default=5, description="返回结果数量")


class RAGTool(BaseTool):
    """
    RAG检索工具

    允许Agent从知识库中检索相关信息
    """
    name: str = "knowledge_search"
    description: str = (
        "从知识库中检索与查询相关的信息。"
        "当需要获取特定领域知识或背景信息时使用此工具。"
        "输入查询文本，返回最相关的知识片段。"
    )
    args_schema: Type[BaseModel] = RAGToolInput

    def __init__(self, vector_store: VectorStore):
        """
        初始化RAG工具

        Args:
            vector_store: 向量存储实例
        """
        super().__init__()
        self._vector_store = vector_store

    def _run(self, query: str, top_k: int = 5) -> str:
        """
        执行检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            str: 格式化的检索结果
        """
        results = self._vector_store.query(query, n_results=top_k)

        # 格式化输出
        if not results['documents'][0]:
            return "未找到相关信息。"

        output = "检索到以下相关信息:\n\n"
        for i, (doc, metadata) in enumerate(
            zip(results['documents'][0], results['metadatas'][0])
        ):
            output += f"【{i+1}】{doc}\n"
            if metadata:
                output += f"   来源: {metadata.get('source', '未知')}\n"
            output += "\n"

        return output
```

#### 3. RAG增强Crew实现 (rag_crew.py)

```python
"""
=============================================================================
阶段四：RAG增强Crew
文件: src/crews/rag_crew.py
说明: 集成RAG能力的多Agent系统
=============================================================================
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List, Optional
from ..rag.vector_store import VectorStore
from ..tools.rag_tool import RAGTool


@CrewBase
class RAGEnhancedCrew:
    """
    RAG增强的AI助手Crew

    特点:
    - 集成知识库检索能力
    - 基于检索结果生成更准确的回答
    - 支持知识库动态更新
    """

    def __init__(
        self,
        knowledge_path: str = "./knowledge",
        collection_name: str = "assistant_kb"
    ):
        """
        初始化RAG增强Crew

        Args:
            knowledge_path: 知识库路径
            collection_name: 向量集合名称
        """
        # 初始化向量存储
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=knowledge_path
        )

        # 创建RAG工具
        self.rag_tool = RAGTool(self.vector_store)

    # =========================================================================
    # Agent 定义
    # =========================================================================

    @agent
    def knowledge_retriever(self) -> Agent:
        """
        知识检索Agent

        职责: 从知识库中检索相关信息
        工具: RAG检索工具
        """
        return Agent(
            config=self.agents_config['knowledge_retriever'],
            tools=[self.rag_tool],
            verbose=True,
        )

    @agent
    def answer_generator(self) -> Agent:
        """
        回答生成Agent

        职责: 基于检索结果生成回答
        """
        return Agent(
            config=self.agents_config['answer_generator'],
            verbose=True,
        )

    # =========================================================================
    # Task 定义
    # =========================================================================

    @task
    def retrieval_task(self) -> Task:
        """检索任务 - 从知识库获取相关信息"""
        return Task(
            config=self.tasks_config['retrieval_task'],
        )

    @task
    def generation_task(self) -> Task:
        """生成任务 - 基于检索结果生成回答"""
        return Task(
            config=self.tasks_config['generation_task'],
        )

    # =========================================================================
    # Crew 定义
    # =========================================================================

    @crew
    def crew(self) -> Crew:
        """创建RAG增强Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    # =========================================================================
    # 知识库管理
    # =========================================================================

    def add_knowledge(
        self,
        documents: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> None:
        """
        添加知识到知识库

        Args:
            documents: 文档列表
            metadatas: 元数据列表
        """
        self.vector_store.add_documents(documents, metadatas)


# =============================================================================
# 运行入口
# =============================================================================

def run_rag_assistant(
    user_query: str,
    knowledge_path: str = "./knowledge"
) -> str:
    """
    运行RAG增强的AI助手

    Args:
        user_query: 用户查询
        knowledge_path: 知识库路径

    Returns:
        str: AI助手的回答
    """
    inputs = {'user_query': user_query}
    rag_crew = RAGEnhancedCrew(knowledge_path=knowledge_path)
    result = rag_crew.crew().kickoff(inputs=inputs)
    return result.raw
```

---

## 阶段五：MCP 集成协议 (Model Context Protocol)

### 目标
- 理解 MCP 协议规范
- 实现 MCP Server 和 Client
- 集成外部工具和服务

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MCP 集成架构                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     MCP 协议层                                   │  │
│   │                                                                  │  │
│   │  ┌──────────────────────────────────────────────────────────┐   │  │
│   │  │                    MCP Client                             │   │  │
│   │  │                  (AI助手系统)                             │   │  │
│   │  └─────────────────────────┬────────────────────────────────┘   │  │
│   │                            │                                     │  │
│   │              ┌─────────────┼─────────────┐                      │  │
│   │              │             │             │                      │  │
│   │              ▼             ▼             ▼                      │  │
│   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │  │
│   │  │ MCP Server 1 │ │ MCP Server 2 │ │ MCP Server 3 │            │  │
│   │  │  文件系统    │ │   数据库     │ │   API服务    │            │  │
│   │  └──────────────┘ └──────────────┘ └──────────────┘            │  │
│   │                                                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     MCP 消息类型                                 │  │
│   │                                                                  │  │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────┐                │  │
│   │  │   Tools    │  │ Resources  │  │  Prompts   │                │  │
│   │  │   工具     │  │   资源     │  │   提示     │                │  │
│   │  └────────────┘  └────────────┘  └────────────┘                │  │
│   │                                                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
phase5_mcp_integration/
├── src/
│   ├── __init__.py
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py           # MCP Server实现
│   │   ├── client.py           # MCP Client实现
│   │   ├── tools/              # MCP工具定义
│   │   │   ├── __init__.py
│   │   │   ├── file_tool.py
│   │   │   └── api_tool.py
│   │   └── resources/          # MCP资源定义
│   │       ├── __init__.py
│   │       └── knowledge_resource.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── mcp_agent.py        # MCP集成Agent
│   ├── config/
│   │   ├── agents.yaml
│   │   ├── tasks.yaml
│   │   └── mcp_config.yaml     # MCP配置
│   └── main.py
├── tests/
│   └── test_mcp.py
└── README.md
```

### 代码示例

#### 1. MCP Server实现 (server.py)

```python
"""
=============================================================================
阶段五：MCP Server实现
文件: src/mcp/server.py
说明: 实现MCP协议的服务端，提供工具和资源
=============================================================================
"""

from mcp.server import Server
from mcp.types import Tool, Resource, TextContent
from typing import List, Dict, Any
import json


class MCPAssistantServer:
    """
    MCP服务端

    提供AI助手所需的工具和资源访问能力
    """

    def __init__(self, name: str = "assistant-server"):
        """
        初始化MCP服务端

        Args:
            name: 服务器名称
        """
        self.server = Server(name)
        self._setup_handlers()

    def _setup_handlers(self):
        """设置消息处理器"""

        # 工具列表处理器
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="search_knowledge",
                    description="搜索知识库中的相关信息",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索查询"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="execute_task",
                    description="执行指定的任务",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_type": {"type": "string"},
                            "parameters": {"type": "object"}
                        },
                        "required": ["task_type"]
                    }
                )
            ]

        # 工具调用处理器
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict) -> List[TextContent]:
            if name == "search_knowledge":
                result = await self._search_knowledge(arguments["query"])
            elif name == "execute_task":
                result = await self._execute_task(
                    arguments["task_type"],
                    arguments.get("parameters", {})
                )
            else:
                result = f"未知工具: {name}"

            return [TextContent(type="text", text=result)]

    async def _search_knowledge(self, query: str) -> str:
        """知识搜索实现"""
        # 实际实现中连接向量数据库
        return f"搜索结果: {query}"

    async def _execute_task(self, task_type: str, params: Dict) -> str:
        """任务执行实现"""
        return f"执行任务: {task_type}, 参数: {params}"

    async def run(self):
        """运行服务器"""
        await self.server.run()
```

#### 2. MCP Client实现 (client.py)

```python
"""
=============================================================================
阶段五：MCP Client实现
文件: src/mcp/client.py
说明: 实现MCP协议的客户端，连接外部服务
=============================================================================
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, Optional
import asyncio


class MCPClient:
    """
    MCP客户端

    连接MCP服务器，调用远程工具和资源
    """

    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}

    async def connect(
        self,
        server_name: str,
        command: str,
        args: List[str] = None
    ) -> None:
        """
        连接到MCP服务器

        Args:
            server_name: 服务器标识名
            command: 启动命令
            args: 命令参数
        """
        server_params = StdioServerParameters(
            command=command,
            args=args or []
        )

        stdio_transport = await stdio_client(server_params)
        read, write = stdio_transport
        session = ClientSession(read, write)

        await session.initialize()
        self.sessions[server_name] = session

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict
    ) -> str:
        """
        调用远程工具

        Args:
            server_name: 服务器名
            tool_name: 工具名
            arguments: 参数

        Returns:
            str: 工具执行结果
        """
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"未连接服务器: {server_name}")

        result = await session.call_tool(tool_name, arguments)
        return result.content[0].text

    async def close(self, server_name: str = None):
        """关闭连接"""
        if server_name:
            if server_name in self.sessions:
                await self.sessions[server_name].close()
        else:
            for session in self.sessions.values():
                await session.close()
```

---

## 阶段六：混合多智能体决策系统 (Hybrid Multi-Agent Decision System)

### 目标
- 整合前五个阶段的所有能力
- 实现智能路由和决策机制
- 构建完整的生产级AI助手系统

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     混合多智能体决策系统架构                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         用户输入层                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                      用户请求                                    │  │ │
│  │  └─────────────────────────────┬───────────────────────────────────┘  │ │
│  └────────────────────────────────┼──────────────────────────────────────┘ │
│                                   ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         智能路由层                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    Router Agent                                  │  │ │
│  │  │                    (意图识别 + 任务分类)                         │  │ │
│  │  └─────────────────────────────┬───────────────────────────────────┘  │ │
│  └────────────────────────────────┼──────────────────────────────────────┘ │
│                                   │                                         │
│              ┌────────────────────┼────────────────────┐                   │
│              ▼                    ▼                    ▼                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         专家Agent层                                    │ │
│  │                                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │ │
│  │  │  知识问答    │  │  任务执行    │  │  创意生成    │                │ │
│  │  │  RAG Agent   │  │  Task Agent  │  │ Creative     │                │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                   │                                         │
│                                   ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         质量保障层                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    RCI 自我反思模块                              │  │ │
│  │  │              (批评 → 改进 → 验证 循环)                          │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                   │                                         │
│                                   ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         外部集成层                                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │ │
│  │  │  MCP Server  │  │  MCP Server  │  │  MCP Server  │                │ │
│  │  │  (文件系统)  │  │  (数据库)    │  │  (API服务)   │                │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
phase6_hybrid_system/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── router.py           # 智能路由器
│   │   ├── orchestrator.py     # 编排器
│   │   └── state_manager.py    # 状态管理
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── router_agent.py     # 路由Agent
│   │   ├── qa_agent.py         # 问答Agent
│   │   ├── task_agent.py       # 任务Agent
│   │   └── creative_agent.py   # 创意Agent
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── rci/                # RCI模块
│   │   ├── rag/                # RAG模块
│   │   └── mcp/                # MCP模块
│   ├── config/
│   │   ├── agents.yaml
│   │   ├── tasks.yaml
│   │   ├── routing_rules.yaml  # 路由规则
│   │   └── system_config.yaml  # 系统配置
│   └── main.py
├── tests/
│   ├── test_router.py
│   ├── test_integration.py
│   └── test_e2e.py
└── README.md
```

### 代码示例

#### 1. 智能路由器 (router.py)

```python
"""
=============================================================================
阶段六：智能路由器
文件: src/core/router.py
说明: 实现请求分类和智能路由
=============================================================================
"""

from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum


class IntentType(Enum):
    """意图类型枚举"""
    KNOWLEDGE_QA = "knowledge_qa"      # 知识问答
    TASK_EXECUTION = "task_execution"  # 任务执行
    CREATIVE_GENERATION = "creative"   # 创意生成
    GENERAL_CHAT = "general_chat"      # 闲聊


class RoutingDecision(BaseModel):
    """路由决策模型"""
    intent: IntentType = Field(description="识别的意图类型")
    confidence: float = Field(ge=0, le=1, description="置信度")
    target_agent: str = Field(description="目标Agent")
    requires_rag: bool = Field(default=False, description="是否需要RAG")
    requires_rci: bool = Field(default=False, description="是否需要RCI")


class IntentRouter:
    """
    意图路由器

    负责分析用户请求并决定路由策略
    """

    def __init__(self):
        self.routing_rules = {
            IntentType.KNOWLEDGE_QA: "qa_agent",
            IntentType.TASK_EXECUTION: "task_agent",
            IntentType.CREATIVE_GENERATION: "creative_agent",
            IntentType.GENERAL_CHAT: "chat_agent"
        }

    def route(self, user_input: str, context: dict = None) -> RoutingDecision:
        """
        路由决策

        Args:
            user_input: 用户输入
            context: 上下文信息

        Returns:
            RoutingDecision: 路由决策
        """
        # 实际实现中使用LLM进行意图识别
        intent = self._classify_intent(user_input)

        return RoutingDecision(
            intent=intent,
            confidence=0.9,
            target_agent=self.routing_rules[intent],
            requires_rag=intent == IntentType.KNOWLEDGE_QA,
            requires_rci=intent in [IntentType.CREATIVE_GENERATION, IntentType.TASK_EXECUTION]
        )

    def _classify_intent(self, text: str) -> IntentType:
        """意图分类（简化实现）"""
        keywords = {
            IntentType.KNOWLEDGE_QA: ["什么是", "如何", "为什么", "解释"],
            IntentType.TASK_EXECUTION: ["帮我", "执行", "创建", "生成文件"],
            IntentType.CREATIVE_GENERATION: ["写一篇", "创作", "设计", "想象"]
        }

        for intent, words in keywords.items():
            if any(w in text for w in words):
                return intent
        return IntentType.GENERAL_CHAT
```

#### 2. 系统编排器 (orchestrator.py)

```python
"""
=============================================================================
阶段六：系统编排器
文件: src/core/orchestrator.py
说明: 协调各模块工作，实现完整的请求处理流程
=============================================================================
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import Dict, Any, Optional
from .router import IntentRouter, RoutingDecision
from ..modules.rci.rci_loop import RCICrew
from ..modules.rag.vector_store import VectorStore
from ..modules.mcp.client import MCPClient


class HybridAssistantOrchestrator:
    """
    混合AI助手编排器

    整合所有模块，提供统一的请求处理接口
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化编排器

        Args:
            config: 系统配置
        """
        # 初始化路由器
        self.router = IntentRouter()

        # 初始化RAG模块
        self.vector_store = VectorStore(
            collection_name=config.get("rag_collection", "knowledge"),
            persist_directory=config.get("rag_path", "./knowledge")
        )

        # 初始化RCI模块
        self.rci_enabled = config.get("rci_enabled", True)
        self.rci_threshold = config.get("rci_threshold", 7.0)
        self.rci_max_iterations = config.get("rci_max_iterations", 3)

        # 初始化MCP客户端
        self.mcp_client = MCPClient()
        self.mcp_servers = config.get("mcp_servers", [])

    async def initialize(self):
        """初始化异步组件"""
        # 连接MCP服务器
        for server in self.mcp_servers:
            await self.mcp_client.connect(
                server["name"],
                server["command"],
                server.get("args", [])
            )

    async def process_request(
        self,
        user_input: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        处理用户请求

        完整流程:
        1. 路由决策
        2. RAG检索（如需要）
        3. Agent执行
        4. RCI优化（如需要）
        5. 返回结果

        Args:
            user_input: 用户输入
            context: 上下文信息

        Returns:
            Dict: 处理结果
        """
        # 1. 路由决策
        routing = self.router.route(user_input, context)

        # 2. RAG检索
        rag_context = ""
        if routing.requires_rag:
            rag_results = self.vector_store.query(user_input, n_results=5)
            rag_context = self._format_rag_results(rag_results)

        # 3. 构建输入
        inputs = {
            "user_input": user_input,
            "rag_context": rag_context,
            "routing_info": routing.dict()
        }

        # 4. 执行Agent
        result = await self._execute_agent(routing.target_agent, inputs)

        # 5. RCI优化
        if routing.requires_rci and self.rci_enabled:
            result = await self._apply_rci(result, user_input)

        return {
            "response": result,
            "routing": routing.dict(),
            "rag_used": routing.requires_rag,
            "rci_applied": routing.requires_rci and self.rci_enabled
        }

    def _format_rag_results(self, results: Dict) -> str:
        """格式化RAG检索结果"""
        if not results['documents'][0]:
            return ""

        formatted = "相关知识:\n"
        for doc in results['documents'][0]:
            formatted += f"- {doc}\n"
        return formatted

    async def _execute_agent(self, agent_name: str, inputs: Dict) -> str:
        """执行指定Agent"""
        # 实际实现中根据agent_name选择对应的Crew
        pass

    async def _apply_rci(self, content: str, original_request: str) -> str:
        """应用RCI优化"""
        rci_crew = RCICrew(
            max_iterations=self.rci_max_iterations,
            threshold=self.rci_threshold
        )
        result = rci_crew.run_rci_loop(original_request)
        return result['final_output']
```

#### 3. 主入口 (main.py)

```python
"""
=============================================================================
阶段六：系统主入口
文件: src/main.py
说明: 混合多智能体AI助手系统的启动入口
=============================================================================
"""

import asyncio
from typing import Optional
from .core.orchestrator import HybridAssistantOrchestrator


# 系统配置
DEFAULT_CONFIG = {
    "rag_collection": "assistant_knowledge",
    "rag_path": "./knowledge",
    "rci_enabled": True,
    "rci_threshold": 7.0,
    "rci_max_iterations": 3,
    "mcp_servers": []
}


async def main():
    """主函数"""
    # 初始化编排器
    orchestrator = HybridAssistantOrchestrator(DEFAULT_CONFIG)
    await orchestrator.initialize()

    print("混合多智能体AI助手已启动")
    print("输入 'quit' 退出\n")

    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() == 'quit':
            break

        result = await orchestrator.process_request(user_input)
        print(f"\n助手: {result['response']}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 依赖配置

### pyproject.toml

```toml
[project]
name = "hybrid-ai-assistant"
version = "1.0.0"
description = "基于CrewAI的混合多智能体AI助手系统"
authors = [{ name = "Your Name", email = "you@example.com" }]
requires-python = ">=3.10,<3.14"

dependencies = [
    # 核心框架
    "crewai[tools]>=1.7.2",

    # RAG相关
    "chromadb>=0.4.0",
    "langchain>=0.1.0",
    "sentence-transformers>=2.2.0",

    # MCP相关
    "mcp>=0.1.0",

    # 工具库
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "aiohttp>=3.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]

[project.scripts]
assistant = "src.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 开发路线总结

### 阶段进度表

| 阶段 | 名称 | 核心内容 | 关键技术 |
|------|------|----------|----------|
| 一 | 基础Agent交互 | 单Agent对话 | CrewAI基础、YAML配置 |
| 二 | 多Agent协作 | 任务链、层级管理 | Sequential/Hierarchical Process |
| 三 | RCI自我反思 | 批评-改进循环 | 质量评估、迭代优化 |
| 四 | RAG知识增强 | 向量检索、知识管理 | ChromaDB、Embedding |
| 五 | MCP集成 | 外部工具、服务连接 | MCP协议、Server/Client |
| 六 | 混合决策系统 | 智能路由、模块整合 | 编排器、状态管理 |

### 技术栈总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           技术栈架构                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  应用层    │  混合多智能体AI助手                                        │
│            │                                                            │
├────────────┼────────────────────────────────────────────────────────────┤
│            │  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  能力层    │  │   RCI    │  │   RAG    │  │   MCP    │                │
│            │  │ 自我反思 │  │ 知识增强 │  │ 外部集成 │                │
│            │  └──────────┘  └──────────┘  └──────────┘                │
│            │                                                            │
├────────────┼────────────────────────────────────────────────────────────┤
│            │  ┌─────────────────────────────────────────────────────┐  │
│  框架层    │  │                    CrewAI                           │  │
│            │  │         (Agent, Task, Crew, Process)                │  │
│            │  └─────────────────────────────────────────────────────┘  │
│            │                                                            │
├────────────┼────────────────────────────────────────────────────────────┤
│            │  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  基础层    │  │ ChromaDB │  │ Pydantic │  │  asyncio │                │
│            │  │ 向量存储 │  │ 数据验证 │  │ 异步处理 │                │
│            │  └──────────┘  └──────────┘  └──────────┘                │
│            │                                                            │
├────────────┼────────────────────────────────────────────────────────────┤
│            │  ┌─────────────────────────────────────────────────────┐  │
│  LLM层     │  │     OpenAI GPT-4 / Claude / 本地模型                │  │
│            │  └─────────────────────────────────────────────────────┘  │
│            │                                                            │
└────────────┴────────────────────────────────────────────────────────────┘
```

### 学习建议

1. **循序渐进**: 按阶段顺序学习，每个阶段都建立在前一阶段的基础上
2. **动手实践**: 每个阶段都有完整的代码示例，建议实际运行和修改
3. **理解原理**: 不仅要会用，更要理解每个模块的设计原理
4. **扩展思考**: 思考如何将这些技术应用到自己的项目中

### 参考资源

- [CrewAI 官方文档](https://docs.crewai.com/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [MCP 协议规范](https://modelcontextprotocol.io/)
- [Pydantic 文档](https://docs.pydantic.dev/)

---

*文档版本: 1.0*
*最后更新: 2025年1月*
