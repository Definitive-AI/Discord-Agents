from typing import TypedDict, Dict, Any, Literal
from enum import Enum
from langchain.agents import AgentType, create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from config import ANTH_API_KEY
from tools import *
from context import *
from prompts import agent1_prompt, agent2_prompt, agent3_prompt, agent4_prompt, agent5_prompt
from langgraph.graph import END, StateGraph
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiscordBotState(TypedDict):
    input: str
    identified_question: str
    question_category: str
    question_details: str
    generated_response: str
    posted_response_details: str
    response_data_for_analytics: str
    updated_knowledge_base: str
    analytics_report: str

class ResponseAction(Enum):
    GENERATE = "generate"
    SKIP = "skip"

llm = ChatAnthropic(temperature=0.3, anthropic_api_key=ANTH_API_KEY, model='claude-3-opus-20240229')

def create_agent_executor(tools: list, prompt: PromptTemplate) -> AgentExecutor:
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

agent1_executor = create_agent_executor([discord_api_tool, knowledge_base_query, discord_api, kbms_api, discord_guidelines_search, question_detection_search], agent1_prompt)
agent2_executor = create_agent_executor([knowledge_base_query, exa_search, exa_content_retrieval, question_detection_search, nlp_qa_search, qa_system_search, discord_bot_response_preparation, discord_guidelines_search, knowledge_management_search, community_engagement_search], agent2_prompt)
agent3_executor = create_agent_executor([discord_api, discord_api_tool], agent3_prompt)
agent4_executor = create_agent_executor([kbms_api, schedule_review, cancel_review, list_reviews, knowledge_management_search], agent4_prompt)
agent5_executor = create_agent_executor([data_analytics_and_reporting, user_feedback, community_engagement, community_engagement_search], agent5_prompt)

def monitoring_identification_node(state: DiscordBotState) -> DiscordBotState:
    try:
        result = agent1_executor.invoke({"input": state["input"]})
        output = result["output"]
        return {
            **state,
            "identified_question": output.get("identified_question", ""),
            "question_category": output.get("question_category", ""),
            "question_details": output.get("question_details", "")
        }
    except Exception as e:
        logger.error(f"Error in monitoring_identification_node: {str(e)}")
        return state

def response_generation_node(state: DiscordBotState) -> DiscordBotState:
    try:
        agent2_input = f"Identified question: {state['identified_question']}\nCategory: {state['question_category']}"
        result = agent2_executor.invoke({"input": agent2_input})
        return {**state, "generated_response": result["output"]}
    except Exception as e:
        logger.error(f"Error in response_generation_node: {str(e)}")
        return state

def response_review_posting_node(state: DiscordBotState) -> DiscordBotState:
    try:
        result = agent3_executor.invoke({"input": state["generated_response"]})
        output = result["output"]
        return {
            **state,
            "posted_response_details": output.get("posted_response", ""),
            "response_data_for_analytics": output.get("analytics_data", "")
        }
    except Exception as e:
        logger.error(f"Error in response_review_posting_node: {str(e)}")
        return state

def knowledge_management_ticketing_node(state: DiscordBotState) -> DiscordBotState:
    try:
        agent4_input = f"Question details: {state['question_details']}\nPosted Response: {state['posted_response_details']}"
        result = agent4_executor.invoke({"input": agent4_input})
        return {**state, "updated_knowledge_base": result["output"]}
    except Exception as e:
        logger.error(f"Error in knowledge_management_ticketing_node: {str(e)}")
        return state

def analytics_community_engagement_node(state: DiscordBotState) -> DiscordBotState:
    try:
        agent5_input = f"Response data: {state['response_data_for_analytics']}\nUpdated knowledge base: {state['updated_knowledge_base']}"
        result = agent5_executor.invoke({"input": agent5_input})
        return {**state, "analytics_report": result["output"]}
    except Exception as e:
        logger.error(f"Error in analytics_community_engagement_node: {str(e)}")
        return state

def should_generate_response(state: DiscordBotState) -> ResponseAction:
    return ResponseAction.GENERATE if state["identified_question"] else ResponseAction.SKIP

workflow = StateGraph(DiscordBotState)
workflow.add_node("monitoring_identification", monitoring_identification_node)
workflow.add_node("response_generation", response_generation_node)
workflow.add_node("response_review_posting", response_review_posting_node)
workflow.add_node("knowledge_management_ticketing", knowledge_management_ticketing_node)
workflow.add_node("analytics_community_engagement", analytics_community_engagement_node)

workflow.set_entry_point("monitoring_identification")
workflow.add_conditional_edges(
    "monitoring_identification",
    should_generate_response,
    {
        ResponseAction.GENERATE: "response_generation",
        ResponseAction.SKIP: "analytics_community_engagement"
    }
)
workflow.add_edge("response_generation", "response_review_posting")
workflow.add_edge("response_review_posting", "knowledge_management_ticketing")
workflow.add_edge("knowledge_management_ticketing", "analytics_community_engagement")
workflow.add_edge("analytics_community_engagement", END)

graph = workflow.compile()

async def process_new_message(message: Dict[str, str]) -> str:
    initial_state = DiscordBotState(
        input=message.get('content', ''),
        identified_question="",
        question_category="",
        question_details="",
        generated_response="",
        posted_response_details="",
        response_data_for_analytics="",
        updated_knowledge_base="",
        analytics_report=""
    )
    final_state = await graph.ainvoke(initial_state)
    return final_state["analytics_report"]

async def monitor_discord_server():
    while True:
        try:
            new_messages = await discord_api_tool.aget_new_messages()
            tasks = [process_new_message(message) for message in new_messages]
            results = await asyncio.gather(*tasks)
            for analytics_report in results:
                logger.info(f"Analytics Report: {analytics_report}")
        except Exception as e:
            logger.error(f"Error in monitor_discord_server: {str(e)}")
        await asyncio.sleep(1)  # Add a small delay to prevent excessive API calls

if __name__ == "__main__":
    logger.info("Starting AI Agent Chain...")
    asyncio.run(monitor_discord_server())
