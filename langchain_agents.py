# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Validate environment variables
required_env_vars = ['ANTH_APIKEY', 'DISCORD_API_KEY']
for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Missing required environment variable: {var}")

ANTH_API_KEY = os.environ['ANTH_APIKEY']
DISCORD_API_KEY = os.environ['DISCORD_API_KEY']
# agents.py
from langchain.agents import AgentType, create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from config import ANTH_API_KEY
from tools import *
from context import *
from prompts import agent1_prompt, agent2_prompt, agent3_prompt, agent4_prompt, agent5_prompt

llm = ChatAnthropic(temperature=0.3, anthropic_api_key=ANTH_API_KEY, model='claude-3-opus-20240229')

def create_agent_executor(tools: list, prompt: PromptTemplate) -> AgentExecutor:
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

agent1_executor = create_agent_executor([discord_api_tool, knowledge_base_query, discord_api, kbms_api, discord_guidelines_search, question_detection_search], agent1_prompt)
agent2_executor = create_agent_executor([knowledge_base_query, exa_search, exa_content_retrieval, question_detection_search, nlp_qa_search, qa_system_search, discord_bot_response_preparation, discord_guidelines_search, knowledge_management_search, community_engagement_search], agent2_prompt)
agent3_executor = create_agent_executor([discord_api, discord_api_tool], agent3_prompt)
agent4_executor = create_agent_executor([kbms_api, schedule_review, cancel_review, list_reviews, knowledge_management_search], agent4_prompt)
agent5_executor = create_agent_executor([data_analytics_and_reporting, user_feedback, community_engagement, community_engagement_search], agent5_prompt)
# database.py
import sqlite3
from typing import List, Dict

class Database:
    def __init__(self, db_name: str = 'discord_bot.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    content TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def add_processed_message(self, message_id: str, content: str):
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO processed_messages (message_id, content) VALUES (?, ?)',
                              (message_id, content))

    def get_processed_messages(self) -> List[Dict[str, str]]:
        with self.conn:
            cursor = self.conn.execute('SELECT message_id, content FROM processed_messages')
            return [{'message_id': row[0], 'content': row[1]} for row in cursor.fetchall()]

db = Database()
# discord_bot.py
import asyncio
import logging
from typing import Dict, Any
from agents import agent1_executor, agent2_executor, agent3_executor, agent4_executor, agent5_executor
from tools import discord_api_tool
from database import db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_agent(executor: AgentExecutor, input_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return await executor.ainvoke(input_data)
    except Exception as e:
        logger.error(f"Error processing agent: {str(e)}")
        return {"output": f"Error: {str(e)}"}

async def process_new_message(message: Dict[str, str]) -> str:
    # Agent 1: Monitoring and Identification
    output1 = await process_agent(agent1_executor, {"input": message['content']})
    
    identified_question = output1["output"].get("identified_question", "")
    question_category = output1["output"].get("question_category", "")
    question_details = output1["output"].get("question_details", "")
    
    # Agent 2: Response Generation
    agent2_input = f"Identified question: {identified_question}\nCategory: {question_category}"
    output2 = await process_agent(agent2_executor, {"input": agent2_input})
    generated_response = output2["output"]
    
    # Agent 3: Response Review and Posting
    output3 = await process_agent(agent3_executor, {"input": generated_response})
    posted_response_details = output3["output"].get("posted_response", "")
    response_data_for_analytics = output3["output"].get("analytics_data", "")
    
    # Agent 4: Knowledge Management and Ticketing
    agent4_input = f"Question details: {question_details}\nPosted Response: {posted_response_details}"
    output4 = await process_agent(agent4_executor, {"input": agent4_input})
    updated_knowledge_base = output4["output"]
    
    # Agent 5: Analytics and Community Engagement
    agent5_input = f"Response data: {response_data_for_analytics}\nUpdated knowledge base: {updated_knowledge_base}"
    output5 = await process_agent(agent5_executor, {"input": agent5_input})
    analytics_report = output5["output"]
    
    # Store processed message in the database
    db.add_processed_message(message['id'], message['content'])
    
    return analytics_report

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
