# article-learning

用于论文 **自动推导与批注** 的对抗式多 Agent 框架。两组 Agent 对每个论断争辩；通过对抗的结论会变成结构化批注。

[![CI](https://github.com/wuyouMaster/article-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/wuyouMaster/article-learning/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/article-learning.svg)](https://pypi.org/project/article-learning/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

> 输入一篇论文（Markdown 或 PDF），输出结构化、带置信度等级的批注——每一条都经过对抗式 Agent 的压力测试。

[English README](README.md)

## 特性

- **对抗式验证** — 四类挑战者（逻辑、假设、反例、引文）在命题被接受前进行压力测试
- **流式批注** — 结果即时写入，无需等待整个流程结束
- **结构化输出** — 每条批注都是 Pydantic 模型，包含置信度、推导过程、引文和完整的对抗历史
- **PDF 与 Markdown 输入** — 支持 `.pdf`（通过 `marker-pdf`）或 `.md` 文件
- **可插拔 LLM 后端** — 兼容任何 OpenAI 兼容 API（OpenAI、DeepSeek、通过 vLLM/Ollama 的本地模型）
- **依赖 DAG** — 命题按拓扑序排列；循环依赖通过联合验证处理
- **符号表** — 跨节追踪符号含义，避免同一符号被静语重载
- **完全可测试** — `DeterministicMockLLM` 和 `ScriptedMockLLM` 可在无 API 密钥的情况下运行完整流程

## 架构

```
                         +-----------------------+
                         |     Blackboard        |  <- 单一可信数据源
                         |  (状态机、DAG、      |
                         |   符号表、日志)      |
                         +-----------------------+
                                  ^   ^
                                  |   |
        +-------------------------+   +---------------------------+
        |                                                         |
+-------------------+                                  +----------------------+
|     A 组          |                                  |       B 组           |
| MainAgent (DAG)   |                                  | LogicChallenger      |
| SubAgent  (块级)  |                                  | AssumptionChallenger |
+-------------------+                                  | CounterexampleConst. |
                                                       | CitationChecker      |
                                                       +----------------------+
                                                                  |
                                                                  v
                                                          流式 Annotator
                                                          (现为 JSON / 后续 MCP)
```

### A 组

* **`MainAgent`**：遍历语义块，抽取命题，构建依赖 DAG，维护全局符号表，并按拓扑序调度下一个命题。对互为引用构成的环路（例如互相引用的一组引理）做标记，供后续联合验证。
* **`SubAgent`**：一次负责一个命题。基于原文块写出推导，并回应 B 组的质问。

### B 组（结构化施压，非随机）

| 挑战者 | 职责 |
|--------|------|
| `LogicChallenger` | 查找推导中的无理跳转、「这一步从何而来」 |
| `AssumptionChallenger` | 质疑前提是否真的成立 |
| `CounterexampleConstr.` | 尝试构造具体反例 |
| `CitationChecker` | 核对引用片段是否原文存在且支撑论断 |

编排器每轮轮换上述角色，使压力覆盖不同维度。

### 状态机

```
PENDING -> IN_PROGRESS -> UNDER_CHALLENGE -+-> CONFIRMED
                                           +-> REFUTED
                                           +-> DOUBTFUL
                                           +-> ESCALATED
```

* `consecutive_unbroken_challenges >= soft_pass_streak` → CONFIRMED
* `consecutive_unanswered >= doubt_streak` → DOUBTFUL
* `rounds_completed >= max_rounds` 且未形成明确 streak → ESCALATED

### 置信度等级

| 等级 | 含义 |
|------|------|
| STRONG | 多类挑战者、多轮均通过 |
| WEAK | 已确认，但 streak 较短或挑战类型覆盖不足 |
| DOUBTFUL | A 组未能连续回应，或升级后仍无法裁决 |
| REFUTED | 发现反例或致命逻辑漏洞 |

## 流式批注

`Orchestrator.run(...)` 可挂载任意多个 `Annotator` 输出端。每个命题一旦走出对抗循环会 **立即写入**；流程仍在运行时可用 `tail -f` 查看 JSONL。

未来的 MCP/PDF 批注器只需实现同一协议，核心编排无需改动。

## 配置

所有配置通过环境变量（或 `.env` 文件）加载：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | — | LLM 提供商的 API 密钥 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 模型名称 |
| `OPENAI_BASE_URL` | — | 覆盖为非 OpenAI 提供商（如 DeepSeek） |
| `MAX_ROUNDS_PER_PROPOSITION` | `4` | 每个命题的最大对抗轮数 |
| `SOFT_PASS_STREAK` | `2` | 连续通过轮数达到此值后标记为 CONFIRMED |
| `DOUBT_STREAK` | `2` | 连续未回应轮数达到此值后标记为 DOUBTFUL |
| `ARTICLE_LEARNING_LOG_LEVEL` | `INFO` | 日志级别 |

## 针对设计风险的缓解

| 风险 | 缓解措施 |
|------|----------|
| 幻觉沿链条传播 | 每条命题携带逐字 `SourceCitation`；由 CitationChecker 校验 |
| 符号跨节歧义 | 全局 `SymbolTable` + 按块作用域；子 Agent 换块时重渲染上下文 |
| 引理循环依赖 | `Blackboard.cycles()` 检测；拓扑排序将环路延后处理 |
| 对抗死循环 | `max_rounds_per_proposition`、`soft_pass_streak`、`doubt_streak` 上限 |

## 快速开始

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest                      # 全量测试：mock LLM 端到端
```

使用真实大模型（OpenAI 兼容 HTTP API）：

```bash
cp .env.example .env
# 填写 OPENAI_API_KEY，按需修改 OPENAI_MODEL / OPENAI_BASE_URL（例如 DeepSeek）
python -m article_learning path/to/paper.md --out annotations.jsonl
```

## 代码示例

```python
from article_learning import Orchestrator
from article_learning.annotators import JSONLAnnotator
from article_learning.ingest import PaperLoader
from article_learning.llm import OpenAIClient

paper = PaperLoader().from_text_file("paper.md")
sinks = [JSONLAnnotator("annotations.jsonl")]
final = Orchestrator(OpenAIClient()).run(paper, annotators=sinks)
print(f"Produced {len(final['annotations'])} annotations")
```

## 导入使用案例

### 一键式流程

`run_pipeline` 是一个便捷函数，一次调用完成解析和执行：

```python
from article_learning import run_pipeline
from article_learning.annotators import JSONLAnnotator
from article_learning.llm import OpenAIClient

llm = OpenAIClient(model="gpt-4o")
annotations = run_pipeline(llm, "paper.md", annotators=[JSONLAnnotator("out.jsonl")])

for ann in annotations:
    print(f"{ann.proposition_id}: {ann.confidence.value} — {ann.statement[:80]}")
```

### 检查 Blackboard

运行结束后，`GraphState` 暴露了一个完整填充的 `Blackboard`，包含每个命题的状态、对抗日志和依赖 DAG：

```python
from article_learning import Orchestrator, Blackboard, PropositionStatus
from article_learning.ingest import load_paper
from article_learning.llm import OpenAIClient

paper = load_paper("paper.md")
state = Orchestrator(OpenAIClient()).run(paper)
bb: Blackboard = state["blackboard"]

# 所有已确认的命题
confirmed = bb.by_status(PropositionStatus.CONFIRMED)
print(f"{len(confirmed)} 个命题已确认")

# 遍历依赖 DAG
import networkx as nx
graph: nx.DiGraph = bb.build_graph()
for node in nx.topological_sort(graph):
    prop = bb.get(node)
    print(f"  {node} ({prop.type.value}): {prop.statement[:60]}")

# 检查特定命题的对抗历史
for record in bb.proposition_history("P1"):
    print(f"  第 {record.round_index} 轮: [{record.challenger}] {record.verdict}")
```

### 使用单个模型

每个模型都是 Pydantic `BaseModel`——可以独立构造、序列化和验证：

```python
from article_learning.models import (
    Annotation,
    ConfidenceLevel,
    Proposition,
    PropositionType,
    PropositionStatus,
    SourceCitation,
    Symbol,
    SymbolTable,
)

# 手动创建命题
prop = Proposition(
    proposition_id="P1",
    type=PropositionType.THEOREM,
    statement="If f is continuous on [0,1] then f is bounded.",
    block_id="block-3",
    citations=[SourceCitation(block_id="block-3", quote="f is continuous on [0,1]")],
    depends_on=["P0"],
)

# 符号表：跨节追踪符号含义
st = SymbolTable()
st.add(Symbol(
    name="f",
    description="Real-valued continuous function on [0,1]",
    introduced_in_block="block-1",
    scope_blocks=[],
))
resolved = st.lookup("f", "block-3")
print(resolved.description if resolved else "未知符号")

# 将批注序列化为 JSON
ann = Annotation(
    proposition_id="P1",
    block_id="block-3",
    statement=prop.statement,
    confidence=ConfidenceLevel.STRONG,
    rounds=3,
)
print(ann.model_dump_json(indent=2))
```

### 自定义 Annotator

实现 `Annotator` 协议，将批注写入任意目标（数据库、stdout、WebSocket 等）：

```python
from article_learning.annotators import Annotator
from article_learning.models import Annotation


class PrintAnnotator:
    """最简单的自定义 Annotator：打印到 stdout。"""

    def write(self, annotation: Annotation) -> None:
        icon = annotation.confidence.emoji
        print(f"{icon} {annotation.proposition_id}: {annotation.statement[:80]}")

    def close(self) -> None:
        pass


# 使用
from article_learning import Orchestrator
from article_learning.ingest import load_paper
from article_learning.llm import OpenAIClient

paper = load_paper("paper.md")
Orchestrator(OpenAIClient()).run(paper, annotators=[PrintAnnotator()])
```

### 同时写入 JSONL 和最终 JSON 文件

组合多个 Annotator，同时获得流式输出和单文件摘要：

```python
from article_learning import Orchestrator
from article_learning.annotators import JSONFileAnnotator, JSONLAnnotator
from article_learning.ingest import load_paper
from article_learning.llm import OpenAIClient

paper = load_paper("paper.md")
annotators = [
    JSONLAnnotator("stream.jsonl"),      # 运行时可用 tail -f 查看
    JSONFileAnnotator("annotations.json"), # 关闭时输出单个 JSON 数组
]
Orchestrator(OpenAIClient()).run(paper, annotators=annotators)
```

### 使用 Mock LLM 进行测试 / 开发

`DeterministicMockLLM` 根据 agent 标签分发请求，无需 API 密钥即可运行完整流程：

```python
import json
from article_learning import Orchestrator
from article_learning.ingest import PaperLoader
from article_learning.llm.mock import DeterministicMockLLM

mock = DeterministicMockLLM()

# 按 agent 标签注册处理器
mock.register("main", lambda msgs: json.dumps({
    "propositions": [
        {
            "proposition_id": "P1",
            "type": "theorem",
            "statement": "Every bounded sequence has a convergent subsequence.",
            "formal_statement": None,
            "block_id": "block-0",
            "citation_quote": "bounded sequence ... convergent subsequence",
            "depends_on": [],
        }
    ],
    "symbols": [],
}))

mock.register("sub", lambda msgs: json.dumps({
    "derivation": "By the Bolzano-Weierstrass theorem.",
    "extra_citations": [],
    "notes": None,
}))

# 挑战者：奇数轮提问，偶数轮通过
for tag in ("logic", "assumption", "counterexample", "citation"):
    mock.register(tag, lambda msgs, t=tag: json.dumps({
        "verdict": "no_issue", "question": "", "rationale": f"{t} pass"
    }))

paper = PaperLoader().from_markdown("# Test\nSome math here.")
state = Orchestrator(mock).run(paper)
print(f"批注数量: {len(state['annotations'])}")
```

### 流式输出到 Rich 控制台

`StreamAnnotator` 将 JSON 行写入任意文本流——配合 `rich.console.Console` 实现漂亮的实时输出：

```python
import sys
from article_learning.annotators import StreamAnnotator
from article_learning import Orchestrator
from article_learning.ingest import load_paper
from article_learning.llm import OpenAIClient

paper = load_paper("paper.md")
stream_annotator = StreamAnnotator(sys.stdout)
Orchestrator(OpenAIClient()).run(paper, annotators=[stream_annotator])
```

### 直接访问 LangGraph 工作流

如需完全控制图结构（自定义断点、部分执行、流式单节点输出），可使用 `build_workflow`：

```python
from article_learning.graph import build_workflow, build_initial_state
from article_learning.ingest import load_paper
from article_learning.llm import OpenAIClient

llm = OpenAIClient()
paper = load_paper("paper.md")
workflow = build_workflow(llm, recursion_limit=300)
initial = build_initial_state(paper)

# 逐节点流式输出
for event in workflow.stream(initial, stream_mode="values"):
    annotations = event.get("annotations", [])
    if annotations:
        print(f"本轮获得 {len(annotations)} 条批注")
```

## PDF 输入

安装可选的 `pdf` 附加组件：

```bash
pip install 'article-learning[pdf]'
```

然后 `PaperLoader().from_pdf("paper.pdf")` 会经由 `marker-pdf` 解析。

## 路线图

* 基于 MCP 的 PDF 批注器：直接向源 PDF 写入标注。
* 由 LLM 驱动的语义分块：替代当前基于规则的首次分段。
* 环路引理的联合验证模式。
* 命题变为 DOUBTFUL 时的人机在环检查点。
