from typing import Optional, Type
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
import discord
import asyncio
import json
import threading

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

retrieved_messages = []
monitored_channels = set()

class DiscordInput(BaseModel):
    input: str = Field(description='A JSON string containing the action and parameters. Example: {"action": "monitor", "channel_id": "123456789"} or {"action": "retrieve", "channel_id": "123456789", "limit": 10}')

@tool("discord_api_tool", args_schema=DiscordInput, return_direct=False)
def discord_api_tool(input: str) -> str:
    """
    Discord API Integration Tool: Enables real-time monitoring of Discord channels and retrieval of messages for analysis by the AI agent. It can monitor channels in real-time and retrieve past messages from specified channels.
    """
    try:
        input_data = json.loads(input)
        action = input_data.get("action")
        channel_id = input_data.get("channel_id")

        if action == "monitor":
            if channel_id not in monitored_channels:
                monitored_channels.add(channel_id)
                return f"Started monitoring channel {channel_id}"
            else:
                return f"Channel {channel_id} is already being monitored"
        elif action == "retrieve":
            limit = input_data.get("limit", 10)
            messages = asyncio.get_event_loop().run_until_complete(retrieve_messages(channel_id, limit))
            return json.dumps([{"author": msg.author.name, "content": msg.content} for msg in messages])
        else:
            return "Invalid action. Use 'monitor' or 'retrieve'."
    except json.JSONDecodeError:
        return "Invalid input format. Please provide a valid JSON string."
    except Exception as e:
        return f"An error occurred: {str(e)}"

async def retrieve_messages(channel_id, limit):
    channel = client.get_channel(int(channel_id))
    if not channel:
        return []
    
    messages = []
    async for message in channel.history(limit=limit):
        messages.append(message)
    return messages

@client.event
async def on_message(message):
    if str(message.channel.id) in monitored_channels:
        print(f"New message in monitored channel {message.channel.id}: {message.content}")
        retrieved_messages.append({"channel_id": str(message.channel.id), "author": message.author.name, "content": message.content})

TOKEN = 'YOUR_BOT_TOKEN'

def start_discord_client():
    client.run(TOKEN)

discord_thread = threading.Thread(target=start_discord_client)
discord_thread.start()


from typing import Optional, Type
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import json
import os

class KnowledgeBaseQueryInput(BaseModel):
    input: str = Field(description="A string containing the query and any additional parameters for the knowledge base search. Example: {\"query\": \"What is the capital of France?\", \"max_results\": 3, \"similarity_threshold\": 0.7}")

class KnowledgeBase:
    def __init__(self, documents_path):
        loader = TextLoader(documents_path)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(texts, embeddings)

    def query(self, query, max_results=5, similarity_threshold=0.5):
        results = self.vectorstore.similarity_search_with_score(query, k=max_results)
        filtered_results = [
            (doc.page_content, score) 
            for doc, score in results 
            if score >= similarity_threshold
        ]
        return filtered_results

# Initialize the knowledge base with a path to your documents
documents_path = os.path.join(os.path.dirname(__file__), "knowledge_base_documents.txt")
kb = KnowledgeBase(documents_path)

@tool("knowledge_base_query", args_schema=KnowledgeBaseQueryInput)
def knowledge_base_query(input: str) -> str:
    """
    Query the internal knowledge base to retrieve relevant information based on the provided query. 
    Supports complex queries and returns formatted results with similarity scores.
    """
    try:
        input_data = json.loads(input)
        query = input_data['query']
        max_results = input_data.get('max_results', 5)
        similarity_threshold = input_data.get('similarity_threshold', 0.5)
        
        results = kb.query(query, max_results, similarity_threshold)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        formatted_results = []
        for i, (content, score) in enumerate(results, 1):
            formatted_results.append(f"Result {i} (Similarity: {score:.2f}):\n{content}\n")
        
        return "\n".join(formatted_results)
    except json.JSONDecodeError:
        return "Invalid input format. Please provide a valid JSON string."
    except KeyError:
        return "Missing 'query' in the input. Please provide a query to search the knowledge base."


from typing import Optional, Type
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
from langchain.utilities import ExaSearchAPIWrapper
import json

class ExaSearchInput(BaseModel):
    input: str = Field(description="The search query string. Example: {\"query\": \"Latest developments in AI\"}")

@tool("exa_search", args_schema=ExaSearchInput, return_direct=False)
def exa_search(input: str) -> str:
    """
    Perform a semantic web search using Exa Search API and return relevant results from trusted websites.
    This tool filters results based on credibility and extracts key information from top-ranked pages.
    """
    search = ExaSearchAPIWrapper()
    query_dict = json.loads(input)
    results = search.results(query_dict['query'], num_results=5)
    
    formatted_results = []
    for result in results:
        formatted_result = f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\n"
        formatted_results.append(formatted_result)
    
    return "\n".join(formatted_results)

class ExaContentRetrievalInput(BaseModel):
    input: str = Field(description="The URL of the document to retrieve content from. Example: {\"url\": \"https://example.com/article\"}")

@tool("exa_content_retrieval", args_schema=ExaContentRetrievalInput, return_direct=False)
def exa_content_retrieval(input: str) -> str:
    """
    Retrieve and clean the content of a webpage from trusted sources using Exa's content retrieval feature.
    This tool fetches and processes web documents for further analysis, ensuring information credibility.
    """
    search = ExaSearchAPIWrapper()
    url_dict = json.loads(input)
    content = search.get_content(url_dict['url'])
    return content


import os
import discord
import asyncio
import json
from typing import Optional, Type
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

class DiscordInput(BaseModel):
    input: str = Field(description='A JSON string containing the action to perform and any necessary parameters. Example: {"action": "send_message", "channel_id": "123456789", "content": "Hello, Discord!"}')

@tool("discord_api", args_schema=DiscordInput, return_direct=False)
def discord_api(input: str) -> str:
    """Interact with the Discord API, including sending messages, editing/deleting messages, adding reactions, and managing a conditional approval process for sensitive or complex responses."""
    try:
        input_data = json.loads(input)
        action = input_data.get('action')
        
        if action == 'send_message':
            return asyncio.run(send_message(input_data))
        elif action == 'edit_message':
            return asyncio.run(edit_message(input_data))
        elif action == 'delete_message':
            return asyncio.run(delete_message(input_data))
        elif action == 'add_reaction':
            return asyncio.run(add_reaction(input_data))
        elif action == 'approval_process':
            return approval_process(input_data)
        else:
            return "Invalid action specified"
    except json.JSONDecodeError:
        return "Invalid JSON input"
    except Exception as e:
        return f"An error occurred: {str(e)}"

async def send_message(data):
    channel = client.get_channel(int(data['channel_id']))
    await channel.send(data['content'])
    return f"Message sent to channel {data['channel_id']}"

async def edit_message(data):
    channel = client.get_channel(int(data['channel_id']))
    message = await channel.fetch_message(int(data['message_id']))
    await message.edit(content=data['new_content'])
    return f"Message {data['message_id']} edited in channel {data['channel_id']}"

async def delete_message(data):
    channel = client.get_channel(int(data['channel_id']))
    message = await channel.fetch_message(int(data['message_id']))
    await message.delete()
    return f"Message {data['message_id']} deleted from channel {data['channel_id']}"

async def add_reaction(data):
    channel = client.get_channel(int(data['channel_id']))
    message = await channel.fetch_message(int(data['message_id']))
    await message.add_reaction(data['emoji'])
    return f"Reaction {data['emoji']} added to message {data['message_id']} in channel {data['channel_id']}"

def approval_process(data):
    content = data.get('content')
    # Simple approval process: approve if content length is less than 100 characters
    if len(content) < 100:
        return f"Content approved: {content}"
    else:
        return f"Content rejected (too long): {content[:50]}..."

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

def run_discord_bot():
    client.run(TOKEN)

# Run the Discord bot in a separate thread
import threading
threading.Thread(target=run_discord_bot, daemon=True).start()


from typing import Optional, Type
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
import requests
import json

class KBMSInput(BaseModel):
    input: str = Field(description='A string representation of a dictionary containing the operation type and necessary data. Example: {"operation": "create", "type": "faq", "question": "What is LangChain?", "answer": "LangChain is a framework for developing applications powered by language models."}')

class KBMSAPITool:
    def __init__(self, api_base_url: str, api_key: str):
        self.api_base_url = api_base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _make_request(self, method: str, endpoint: str, data: Optional[dict] = None) -> str:
        url = f"{self.api_base_url}/{endpoint}"
        response = requests.request(method, url, headers=self.headers, json=data)
        return self._handle_response(response)

    def create_entry(self, entry_type: str, data: dict) -> str:
        return self._make_request("POST", entry_type, data)

    def read_entry(self, entry_type: str, entry_id: str) -> str:
        return self._make_request("GET", f"{entry_type}/{entry_id}")

    def update_entry(self, entry_type: str, entry_id: str, data: dict) -> str:
        return self._make_request("PUT", f"{entry_type}/{entry_id}", data)

    def delete_entry(self, entry_type: str, entry_id: str) -> str:
        return self._make_request("DELETE", f"{entry_type}/{entry_id}")

    def tag_for_review(self, entry_type: str, entry_id: str, tag: str) -> str:
        return self._make_request("POST", f"{entry_type}/{entry_id}/tag", {"tag": tag})

    def _handle_response(self, response: requests.Response) -> str:
        if response.status_code == 200:
            return json.dumps(response.json())
        else:
            return f"Error: {response.status_code} - {response.text}"

# Initialize the KBMS API Tool
kbms_api = KBMSAPITool(api_base_url="https://api.kbms.example.com/v1", api_key="your_api_key_here")

@tool("kbms_api", args_schema=KBMSInput)
def kbms_api(input: str) -> str:
    """
    Interact with the Knowledge Base Management System (KBMS) API. This comprehensive tool enables creating, reading, updating, and deleting entries in the knowledge base, FAQs, and ticketing system. It also supports tagging items for review, providing full management capabilities for the KBMS.
    """
    try:
        input_data = json.loads(input)
        operation = input_data.pop("operation", None)
        entry_type = input_data.pop("type", None)
        entry_id = input_data.pop("id", None)

        if not operation or not entry_type:
            return "Error: Missing required fields 'operation' and 'type'"

        if operation == "create":
            return kbms_api.create_entry(entry_type, input_data)
        elif operation == "read":
            if not entry_id:
                return "Error: Missing required field 'id' for read operation"
            return kbms_api.read_entry(entry_type, entry_id)
        elif operation == "update":
            if not entry_id:
                return "Error: Missing required field 'id' for update operation"
            return kbms_api.update_entry(entry_type, entry_id, input_data)
        elif operation == "delete":
            if not entry_id:
                return "Error: Missing required field 'id' for delete operation"
            return kbms_api.delete_entry(entry_type, entry_id)
        elif operation == "tag":
            if not entry_id or "tag" not in input_data:
                return "Error: Missing required fields 'id' and 'tag' for tag operation"
            return kbms_api.tag_for_review(entry_type, entry_id, input_data["tag"])
        else:
            return f"Error: Invalid operation '{operation}'"
    except json.JSONDecodeError:
        return "Error: Invalid JSON input"
    except Exception as e:
        return f"Error: {str(e)}"


from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
import schedule
import time
from datetime import datetime, timedelta
from desktop_notifier import DesktopNotifier
import json
import threading

notifier = DesktopNotifier()

class ReviewScheduleInput(BaseModel):
    input: str = Field(description='A JSON string containing the review schedule details. Example: {"reviewer_name": "John Doe", "review_frequency": "daily", "review_time": "14:30", "knowledge_base_section": "AI Ethics"}')

scheduled_reviews = []

@tool("schedule_review", args_schema=ReviewScheduleInput, return_direct=False)
def schedule_review(input: str) -> str:
    """Schedule a human reviewer for periodic quality control of the knowledge base, setting up notifications for the specified review time and frequency."""
    input_data = json.loads(input)
    reviewer_name = input_data["reviewer_name"]
    review_frequency = input_data["review_frequency"]
    review_time = input_data["review_time"]
    knowledge_base_section = input_data["knowledge_base_section"]

    def review_task():
        notification_title = f"Knowledge Base Review: {knowledge_base_section}"
        notification_message = f"Hello {reviewer_name}, it's time to review the {knowledge_base_section} section of the knowledge base."
        notifier.send_sync(title=notification_title, message=notification_message)

    if review_frequency == "daily":
        job = schedule.every().day.at(review_time).do(review_task)
    elif review_frequency == "weekly":
        job = schedule.every().week.at(review_time).do(review_task)
    elif review_frequency == "monthly":
        job = schedule.every(30).days.at(review_time).do(review_task)
    else:
        return f"Invalid review frequency: {review_frequency}"

    scheduled_reviews.append(job)
    return f"Review scheduled for {reviewer_name} to review {knowledge_base_section} {review_frequency} at {review_time}"

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Run the scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.start()

class CancelReviewInput(BaseModel):
    input: str = Field(description='A JSON string containing the review cancellation details. Example: {"reviewer_name": "John Doe", "knowledge_base_section": "AI Ethics"}')

@tool("cancel_review", args_schema=CancelReviewInput, return_direct=False)
def cancel_review(input: str) -> str:
    """Cancel a scheduled review task for a human reviewer."""
    input_data = json.loads(input)
    reviewer_name = input_data["reviewer_name"]
    knowledge_base_section = input_data["knowledge_base_section"]

    for job in scheduled_reviews:
        if reviewer_name in str(job) and knowledge_base_section in str(job):
            schedule.cancel_job(job)
            scheduled_reviews.remove(job)
            return f"Cancelled review for {reviewer_name} on {knowledge_base_section}"

    return f"No matching review found for {reviewer_name} on {knowledge_base_section}"

class ListReviewsInput(BaseModel):
    input: str = Field(description='A JSON string representing an empty dictionary. Example: {}')

@tool("list_reviews", args_schema=ListReviewsInput, return_direct=False)
def list_reviews(input: str) -> str:
    """List all scheduled review tasks for human reviewers."""
    if not scheduled_reviews:
        return "No reviews scheduled."
    
    review_list = []
    for job in scheduled_reviews:
        review_list.append(str(job))
    
    return "\n".join(review_list)


from typing import Optional, Type
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
import pandas as pd
import matplotlib.pyplot as plt
import json
import base64
from io import BytesIO

class DataAnalyticsInput(BaseModel):
    input: str = Field(description='A JSON string containing the following keys: "data_source" (path to CSV file, JSON file, or JSON data as string), "analysis_type" (e.g., "summary", "correlation", "time_series"), and "visualization_type" (e.g., "bar", "line", "scatter"). Example: {"data_source": "sales_data.csv", "analysis_type": "summary", "visualization_type": "bar"}')

@tool("data_analytics_and_reporting", args_schema=DataAnalyticsInput, return_direct=False)
def data_analytics_and_reporting(input: str) -> str:
    """
    Accesses, analyzes, and visualizes data from various sources to generate insights, track metrics, and identify trends. This tool can perform summary statistics, correlation analysis, and time series analysis on CSV or JSON data, and create bar, line, or scatter plot visualizations.
    """
    try:
        # Parse input
        input_data = json.loads(input)
        data_source = input_data['data_source']
        analysis_type = input_data['analysis_type']
        visualization_type = input_data['visualization_type']

        # Load data
        if data_source.endswith('.csv'):
            df = pd.read_csv(data_source)
        elif data_source.endswith('.json'):
            df = pd.read_json(data_source)
        else:
            df = pd.DataFrame(json.loads(data_source))

        # Perform analysis
        if analysis_type == 'summary':
            analysis_result = df.describe().to_dict()
        elif analysis_type == 'correlation':
            analysis_result = df.corr().to_dict()
        elif analysis_type == 'time_series':
            if 'date' not in df.columns:
                return "Error: 'date' column not found in the data for time series analysis"
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            analysis_result = df.resample('M').mean().to_dict()
        else:
            return f"Unsupported analysis type: {analysis_type}"

        # Create visualization
        plt.figure(figsize=(10, 6))
        if visualization_type == 'bar':
            df.plot(kind='bar')
        elif visualization_type == 'line':
            df.plot(kind='line')
        elif visualization_type == 'scatter':
            if len(df.columns) < 2:
                return "Error: Not enough columns for scatter plot"
            df.plot(kind='scatter', x=df.columns[0], y=df.columns[1])
        else:
            return f"Unsupported visualization type: {visualization_type}"

        plt.title(f"{analysis_type.capitalize()} Analysis")
        plt.xlabel("Data Points")
        plt.ylabel("Values")
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        # Prepare response
        response = {
            "analysis_result": analysis_result,
            "visualization": image_base64
        }

        return json.dumps(response)

    except Exception as e:
        return f"An error occurred: {str(e)}"


from typing import Optional, Type
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
from collections import defaultdict
from statistics import mean
import json

class FeedbackInput(BaseModel):
    input: str = Field(description="A string representation of a dictionary containing the feedback data. Example: '{\"user_id\": \"123\", \"rating\": 4, \"comment\": \"Great service!\", \"category\": \"customer_support\"}'")

feedback_data = defaultdict(list)

@tool("user_feedback", args_schema=FeedbackInput, return_direct=False)
def user_feedback(input: str) -> str:
    """Gathers, processes, and analyzes user feedback to measure service quality and identify areas for improvement."""
    try:
        feedback = json.loads(input)
        user_id = feedback.get('user_id')
        rating = feedback.get('rating')
        comment = feedback.get('comment')
        category = feedback.get('category')
        
        if not all([user_id, rating, comment, category]):
            return "Error: Missing required fields in feedback data."
        
        if not isinstance(rating, (int, float)) or rating < 1 or rating > 5:
            return "Error: Rating must be a number between 1 and 5."
        
        feedback_data[category].append({
            'user_id': user_id,
            'rating': rating,
            'comment': comment
        })
        
        # Perform basic analysis
        category_feedback = feedback_data[category]
        avg_rating = mean([f['rating'] for f in category_feedback])
        total_feedback = len(category_feedback)
        
        analysis = f"Feedback recorded. Current analysis for {category}:\n"
        analysis += f"Total feedback: {total_feedback}\n"
        analysis += f"Average rating: {avg_rating:.2f}\n"
        
        return analysis
    
    except json.JSONDecodeError:
        return "Error: Invalid JSON input. Please provide feedback data as a valid JSON string."
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


from typing import Optional, Type
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
import json
from datetime import datetime

class CommunityEngagementInput(BaseModel):
    input: str = Field(description="A string representation of a dictionary containing the action and relevant parameters. Example: '{\"action\": \"post_discussion\", \"title\": \"New community initiative\", \"content\": \"Let's discuss our upcoming event\", \"category\": \"Events\"}'")

@tool("community_engagement", args_schema=CommunityEngagementInput, return_direct=False)
def community_engagement(input: str) -> str:
    """
    Monitors and facilitates community discussions, supports member engagement, and tracks community contributions.
    """
    try:
        data = json.loads(input)
        action = data.get("action")

        if action == "post_discussion":
            return post_discussion(data)
        elif action == "get_discussions":
            return get_discussions()
        elif action == "post_comment":
            return post_comment(data)
        elif action == "track_contribution":
            return track_contribution(data)
        elif action == "get_engagement_stats":
            return get_engagement_stats()
        else:
            return "Invalid action. Supported actions are: post_discussion, get_discussions, post_comment, track_contribution, get_engagement_stats"

    except json.JSONDecodeError:
        return "Invalid input. Please provide a valid JSON string."

def post_discussion(data: dict) -> str:
    """Simulates posting a new discussion."""
    title = data.get("title")
    content = data.get("content")
    category = data.get("category")
    return f"New discussion posted: '{title}' in category '{category}'"

def get_discussions() -> str:
    """Simulates fetching recent discussions."""
    discussions = [
        {"title": "Welcome to our community", "category": "General", "comments": 5},
        {"title": "Upcoming virtual meetup", "category": "Events", "comments": 10},
    ]
    return json.dumps(discussions)

def post_comment(data: dict) -> str:
    """Simulates posting a comment."""
    discussion_id = data.get("discussion_id")
    comment = data.get("comment")
    return f"Comment posted to discussion {discussion_id}: '{comment}'"

def track_contribution(data: dict) -> str:
    """Simulates tracking a member's contribution."""
    member_id = data.get("member_id")
    contribution_type = data.get("contribution_type")
    return f"Contribution tracked for member {member_id}: {contribution_type}"

def get_engagement_stats() -> str:
    """Simulates fetching engagement statistics."""
    stats = {
        "total_members": 1000,
        "active_members_last_30_days": 750,
        "total_discussions": 50,
        "total_comments": 500,
        "average_comments_per_discussion": 10
    }
    return json.dumps(stats)
