# 多智能体报告生成系统
本文介绍了一个简单的多智能体系统的实现，使用 LangChain 结合 LangGraph 来构建这样的系统，LangGraph 作为 LangChain 生态的重要组成部分，专为构建有状态、多智能体的工作流而设计，非常适合实现复杂的智能体协作场景。

本文将详细介绍如何使用 LangChain 和 LangGraph 构建一个多智能体报告生成系统，该系统能够自动完成从网络研究、趋势分析、报告撰写到校对的全流程。

系统设计概述
我们的多智能体系统将包含四个专业智能体和一个管理智能体，形成一个监督式架构：

网络研究智能体：负责通过网络搜索获取目标主题的最新信息

趋势分析智能体：分析研究结果，提取并排序关键趋势

报告撰写智能体：基于研究和分析结果撰写专业报告

校对智能体：优化报告的语法、格式和可读性

管理智能体：协调各智能体工作，控制整体流程

系统采用状态驱动的工作流，通过 LangGraph 构建状态转换图，实现智能体之间的有序协作。

开发步骤
步骤 1：安装所需库
首先安装实现系统所需的库：

pip install langchain langchain-openai langchain-community langgraph pydantic
步骤 2：导入必要的库
import os

from typing import Dict, List, Any

from pydantic import BaseModel, Field

from langchain\_openai import ChatOpenAI

from langchain\_core.tools import tool

from langchain\_community.tools import BraveSearch

from langchain\_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain\_core.output\_parsers import StrOutputParser

from langchain\_core.messages import BaseMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, END

from langgraph.checkpoint.memory import MemorySaver
步骤 3：配置 API 密钥
本系统使用 OpenAI 的语言模型和 Brave 搜索引擎，需要配置相应的 API 密钥：

\# 设置API密钥

os.environ\["OPENAI\_API\_KEY"] = "YOUR-OPENAI-API-KEY"

os.environ\["BRAVE\_API\_KEY"] = "YOUR-BRAVE-API-KEY"
步骤 4：定义系统状态结构
在多智能体系统中，状态管理至关重要。我们定义一个 AgentState 类来保存整个工作流的状态：

class AgentState(BaseModel):

   """多智能体系统的全局状态"""

   topic: str = Field(description="报告的主题")

   research\_findings: str = Field(default="", description="网络研究的结果")

   trend\_analysis: str = Field(default="", description="趋势分析的结果")

   draft\_report: str = Field(default="", description="报告草稿")

   final\_report: str = Field(default="", description="校对后的最终报告")

   current\_agent: str = Field(default="", description="当前正在工作的智能体")

   next\_agent: str = Field(default="", description="下一个要工作的智能体")

   messages: List\[BaseMessage] = Field(default\_factory=list, description="消息历史")
这个状态类包含了系统运行过程中需要共享和传递的所有信息，确保智能体之间能够有效协作。

步骤 5：定义工具
智能体通过工具与外部环境交互。我们使用 Brave 搜索引擎作为网络研究智能体的工具：

@tool

def brave\_search(query: str) -> str:

   """使用Brave搜索引擎搜索网络上的信息"""

   brave = BraveSearch.from\_api\_key(

       api\_key=os.environ\["BRAVE\_API\_KEY"],

       search\_kwargs={"count": 3}  # 返回3条搜索结果

   )

   return brave.run(query)
通过 LangChain 的 @tool 装饰器，我们可以轻松定义供智能体使用的工具。

步骤 6：设计智能体类
我们创建一个通用的 Agent 类来封装智能体的基本行为，每个智能体将拥有自己的角色、目标、背景故事和可用工具：

class Agent:

   def \_\_init\_\_(self, name: str, role: str, goal: str, backstory: str, tools: List\[Any] = None):

       self.name = name

       self.role = role

       self.goal = goal

       self.backstory = backstory

       self.tools = tools or \[]

       # 创建提示模板

       self.prompt = ChatPromptTemplate.from\_messages(\[

           ("system", f"""你是{self.name}，角色是{self.role}。

           你的目标是：{self.goal}

           你的背景：{self.backstory}

           可用工具：{\[tool.name for tool in self.tools]}

           如果你有可用工具，在需要时可以使用它们。

           请根据当前任务和可用信息，完成你的工作。"""),

           MessagesPlaceholder(variable\_name="messages"),

       ])

       # 创建LLM

       self.llm = ChatOpenAI(model="gpt-4o")

       # 如果有工具，创建带工具调用的链

       if self.tools:

           self.chain = self.prompt | self.llm.bind\_tools(self.tools)

       else:

           self.chain = self.prompt | self.llm | StrOutputParser()

          

   def invoke(self, state: AgentState) -> Dict\[str, Any]:

       """调用智能体处理任务"""

       # 构建智能体的输入消息

       messages = \[

           HumanMessage(content=f"请处理关于{state.topic}的任务。当前可用信息：{self.\_get\_relevant\_info(state)}")

       ]

       # 调用智能体

       result = self.chain.invoke({"messages": messages})

       # 处理结果并更新状态

       return self.\_process\_result(state, result)

   def \_get\_relevant\_info(self, state: AgentState) -> str:

       """获取与当前智能体相关的信息"""

       if self.name == "web\_researcher":

           return f"主题：{state.topic}"

       elif self.name == "trend\_analyst":

           return f"研究结果：{state.research\_findings}"

       elif self.name == "report\_writer":

           return f"研究结果：{state.research\_findings}\n趋势分析：{state.trend\_analysis}"

       elif self.name == "proofreader":

           return f"报告草稿：{state.draft\_report}"

       return ""

   def \_process\_result(self, state: AgentState, result: Any) -> Dict\[str, Any]:

       """处理智能体的输出结果并更新状态"""

       # 提取结果内容

       if hasattr(result, 'content'):

           content = result.content

       elif isinstance(result, str):

           content = result

       else:

           content = str(result)

       # 根据智能体类型更新相应的状态字段

       updates = {

           "current\_agent": self.name,

           "messages": state.messages + \[AIMessage(content=content, name=self.name)]

       }

       # 设置下一步智能体

       if self.name == "web\_researcher":

           updates\["research\_findings"] = content

           updates\["next\_agent"] = "trend\_analyst"

       elif self.name == "trend\_analyst":

           updates\["trend\_analysis"] = content

           updates\["next\_agent"] = "report\_writer"

       elif self.name == "report\_writer":

           updates\["draft\_report"] = content

           updates\["next\_agent"] = "proofreader"

       elif self.name == "proofreader":

           updates\["final\_report"] = content

           updates\["next\_agent"] = ""  # 所有任务完成

       return updates
这个通用智能体类实现了智能体的核心功能：根据提示处理任务、使用工具（如果有）、处理结果并更新系统状态。

步骤 7：创建智能体实例
基于上述 Agent 类，我们创建四个专业智能体：

\# 网络研究智能体

web\_researcher = Agent(

   name="web\_researcher",

   role="网络研究专家",

   goal="找到关于{topic}的最新、有影响力且相关的信息，包括关键用例、挑战和统计数据",

   backstory="你曾是一名调查记者，擅长发现技术突破和市场洞察，能识别可操作的数据和趋势",

   tools=\[brave\_search]  # 网络研究智能体可使用搜索工具

)

\# 趋势分析智能体

trend\_analyst = Agent(

   name="trend\_analyst",

   role="趋势分析专家",

   goal="分析研究结果，提取重要趋势，并按行业影响力、增长潜力和独特性排序",

   backstory="你是一名资深战略顾问，擅长从数据中发现模式，将原始数据转化为清晰的见解"

)

\# 报告撰写智能体

report\_writer = Agent(

   name="report\_writer",

   role="报告撰写专家",

   goal="撰写详细、专业的报告，有效传达研究结果和分析",

   backstory="你曾是知名期刊的技术作家，擅长将故事与数据结合，创作既信息丰富又引人入胜的内容"

)

\# 校对智能体

proofreader = Agent(

   name="proofreader",

   role="校对专家",

   goal="优化报告的语法准确性、可读性和格式，确保符合专业出版标准",

   backstory="你是一位获奖编辑，擅长完善书面内容，对细节有敏锐的洞察力"

)
步骤 8：创建管理智能体
管理智能体负责协调各智能体工作，决定工作流程：

class ManagerAgent:

   def \_\_init\_\_(self):

       self.prompt = ChatPromptTemplate.from\_messages(\[

           ("system", """你是多智能体系统的管理器，负责协调各个智能体完成任务。

           你的职责是根据当前状态决定下一步应该由哪个智能体工作。

           工作流程应该是：web\_researcher → trend\_analyst → report\_writer → proofreader → 完成。

           如果next\_agent已设置，则使用该值。

           只需返回智能体名称，不要添加其他内容。"""),

           MessagesPlaceholder(variable\_name="messages"),

       ])

       self.llm = ChatOpenAI(model="gpt-4o")

       self.chain = self.prompt | self.llm | StrOutputParser()

   def decide\_next\_agent(self, state: AgentState) -> str:

       """决定下一个要工作的智能体"""

       if state.next\_agent:

           return state.next\_agent

       # 如果没有设置next\_agent，则由管理器决定

       decision = self.chain.invoke({

           "messages": \[HumanMessage(content=f"当前状态: {state.dict()}")]

       })

       return decision.strip()
管理智能体根据系统当前状态决定下一步由哪个智能体执行任务，确保工作流程按预定顺序进行。

步骤 9：构建工作流图
使用 LangGraph 构建状态转换图，定义智能体之间的协作流程：

\# 初始化管理智能体

manager = ManagerAgent()

\# 定义图中的节点函数

def web\_researcher\_node(state: AgentState) -> Dict\[str, Any]:

   return web\_researcher.invoke(state)

def trend\_analyst\_node(state: AgentState) -> Dict\[str, Any]:

   return trend\_analyst.invoke(state)

def report\_writer\_node(state: AgentState) -> Dict\[str, Any]:

   return report\_writer.invoke(state)

def proofreader\_node(state: AgentState) -> Dict\[str, Any]:

   return proofreader.invoke(state)

def router(state: AgentState) -> str:

   """路由函数，决定下一步走向"""

   next\_agent = manager.decide\_next\_agent(state)

   if not next\_agent:

       return END

   return next\_agent

\# 构建工作流图

def create\_workflow\_graph():

   # 初始化状态图

   workflow = StateGraph(AgentState)

   # 添加节点

   workflow.add\_node("web\_researcher", web\_researcher\_node)

   workflow.add\_node("trend\_analyst", trend\_analyst\_node)

   workflow.add\_node("report\_writer", report\_writer\_node)

   workflow.add\_node("proofreader", proofreader\_node)

   # 设置入口点

   workflow.set\_entry\_point("web\_researcher")

   # 添加边缘

   workflow.add\_edge("web\_researcher", router)

   workflow.add\_edge("trend\_analyst", router)

   workflow.add\_edge("report\_writer", router)

   workflow.add\_edge("proofreader", router)

   # 编译图，添加内存支持

   memory = MemorySaver()

   return workflow.compile(checkpointer=memory)
在这个工作流图中，我们定义了四个智能体节点和一个路由节点。路由节点由管理智能体控制，决定流程的下一步走向。

步骤 10：运行多智能体系统
最后，我们编写代码来运行整个多智能体系统：

if \_\_name\_\_ == "\_\_main\_\_":

   # 创建工作流

   app = create\_workflow\_graph()

   # 定义报告主题

   topic = "2024年人工智能发展趋势"

   # 初始化状态

   initial\_state = AgentState(

       topic=topic,

       messages=\[HumanMessage(content=f"请生成关于{topic}的报告")]

   )

   # 运行工作流

   final\_state = None

   for step in app.stream(initial\_state, {"configurable": {"thread\_id": "report-1"}}):

       # 打印每个步骤的结果

       for node, values in step.items():

           print(f"\n--- {node} 完成 ---")

           if node == "web\_researcher":

               print(f"研究结果: {values\['research\_findings']\[:200]}...")

           elif node == "trend\_analyst":

               print(f"趋势分析: {values\['trend\_analysis']\[:200]}...")

           elif node == "report\_writer":

               print(f"报告草稿: {values\['draft\_report']\[:200]}...")

           elif node == "proofreader":

               print(f"最终报告: {values\['final\_report']\[:200]}...")

       final\_state = values

   # 输出最终报告

   if final\_state:

       print("\n\n=== 最终报告 ===")

       print(final\_state\["final\_report"])
系统工作流程说明
该多智能体系统按照以下流程运行：

初始化：设置报告主题并创建初始状态

网络研究：网络研究智能体使用 Brave 搜索引擎收集关于主题的信息

趋势分析：趋势分析智能体分析研究结果，提取并排序关键趋势

报告撰写：报告撰写智能体基于研究和分析结果撰写报告草稿

校对优化：校对智能体优化报告的语法、格式和可读性

完成：输出最终报告

每个步骤完成后，管理智能体决定下一步由哪个智能体执行任务，直到所有任务完成。

总结
本文展示了如何使用 LangChain 和 LangGraph 构建多智能体系统，通过 LangGraph 的状态图机制，我们可以清晰地定义智能体之间的协作关系和工作流程，同时保持系统状态的一致性和可追踪性。这种架构不仅适用于报告生成，还可以扩展到其他需要多智能体协作的复杂任务场景。

你可以根据实际需求进一步扩展这个系统，例如添加更多类型的智能体、增加更复杂的路由逻辑，或者集成更多的外部工具。