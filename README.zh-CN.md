# article-learning

用于论文 **自动推导与批注** 的对抗式多 Agent 框架。两组 Agent 对每个论断争辩；通过对抗的结论会变成结构化批注。

[English README](README.md)

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
