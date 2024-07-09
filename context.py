from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores.chroma import Chroma
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings

modelPath = "BAAI/bge-small-en-v1.5" 
model_kwargs = {'device':'cpu','trust_remote_code':'True'}
encode_kwargs = {'normalize_embeddings': True}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

chroma = chromadb.PersistentClient(path="./chroma_db")

from chromadb.api import AdminAPI, ClientAPI
def collection_exists(client:ClientAPI, collection_name):
    collections = client.list_collections()
    filtered_collection = filter(lambda collection: collection.name == collection_name, collections)
    found = any(filtered_collection)
    return found

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import uuid

modelPath = "BAAI/bge-small-en-v1.5" 
model_kwargs = {'device':'cpu','trust_remote_code':'True'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

chroma = chromadb.PersistentClient(path="./chroma_db")

def collection_exists(client, collection_name):
    collections = client.list_collections()
    return collection_name in [collection.name for collection in collections]

def load_documentation(urls, collection_name, tool_name, tool_description):
    if not collection_exists(chroma, collection_name):
        loader = WebBaseLoader(urls)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        
        collection = chroma.create_collection(collection_name)
        for doc in split_docs:
            emb = embeddings.embed_documents([doc.page_content])
            collection.add(
                ids=[str(uuid.uuid1())], 
                embeddings=emb, 
                metadatas=doc.metadata, 
                documents=[doc.page_content]
            )

    retriever = Chroma(client=chroma, collection_name=collection_name, embedding_function=embeddings).as_retriever()
    return create_retriever_tool(
        retriever,
        tool_name,
        tool_description
    )

# Question Detection Tool
question_detection_urls = [
    "https://discord.com/developers/docs/intro",
    "https://discord.js.org/",
    "https://ai.google.com/research/NaturalQuestions/dataset",
    "https://huggingface.co/datasets/natural_questions",
    "https://paperswithcode.com/area/natural-language-processing/question-answering",
    "https://github.com/clulab/nlp-reading-group/wiki/Question-answering-resources"
]

question_detection_tool = load_documentation(
    question_detection_urls,
    "question_detection",
    "question_detection_search",
    "Search for information about question detection and categorization. Use this tool for any questions related to identifying and categorizing questions within a given context!"
)

# NLP QA Tool
nlp_qa_urls = [
    "https://www.datasciencecentral.com/question-answering-tutorial-with-hugging-face-bert/",
    "https://towardsdatascience.com/natural-language-understanding-core-component-of-conversational-agent-3e51357ca934",
    "https://thinkinfi.com/complete-guide-for-natural-language-processing-in-python/",
    "https://rightclick.ai/basic-guide-to-natural-language-processing/",
    "https://www.analytixlabs.co.in/blog/nlp-interview-questions",
    "https://blog.questgen.ai/complete-guide-to-generating-multiple-choice-questions-automatically-using-ai-ckxnbk4l1647361ks7wohnp8x4/"
]

nlp_qa_tool = load_documentation(
    nlp_qa_urls,
    "nlp_qa",
    "nlp_qa_search",
    "Search for information about Natural Language Processing, Question Answering, and related topics. Use this tool for questions about NLP techniques, QA systems, and AI-based text processing."
)

# QA System Tool
qa_system_urls = [
    "https://www.aclweb.org/anthology/N16-4003.pdf",
    "https://web.stanford.edu/~jurafsky/slp3/23.pdf",
    "https://uclnlp.github.io/ai4exams/eqa.html",
    "https://sisinflab.github.io/interactive-question-answering-systems-survey/",
    "https://www.slideshare.net/shekarpour/tutorial-on-question-answering-systems",
    "https://www.microsoft.com/en-us/research/publication/question-answering-with-knowledge-base-web-and-beyond/",
    "https://github.com/clulab/nlp-reading-group/wiki/Question-answering-resources",
    "https://thesai.org/Downloads/Volume12No3/Paper_59-Question_Answering_Systems.pdf"
]

qa_system_tool = load_documentation(
    qa_system_urls,
    "qa_system",
    "qa_system_search",
    "Search for information about question answering systems. Use this tool for queries related to creating or retrieving appropriate responses, utilizing knowledge bases and external resources."
)

# Discord Bot Response Tool
discord_bot_urls = [
    "https://github.com/camdan-me/DiscordBotBestPractices",
    "https://github.com/andrelucaas/discord-bot-best-practices",
    "https://discord.com/developers/docs/intro",
    "https://v13.discordjs.guide/",
    "https://botblock.org/lists/best-practices",
    "https://www.toptal.com/chatbot/how-to-make-a-discord-bot"
]

discord_bot_tool = load_documentation(
    discord_bot_urls,
    "discord_bot",
    "discord_bot_response_preparation",
    "Search for information about Discord bot response preparation, formatting, and posting guidelines. Use this tool for questions related to ensuring quality, consistency, and appropriateness of responses before posting to the Discord server."
)

# Discord Guidelines Tool
discord_guidelines_urls = [
    "https://discord.com/guidelines",
    "https://support.discord.com/hc/en-us/articles/360035969312",
    "https://support.discord.com/hc/en-us/articles/115001987272",
    "https://discord.com/moderation",
    "https://discord.com/community/keeping-your-community-safe",
    "https://github.com/wc3717/Discord-Server-Best-Practices"
]

discord_guidelines_tool = load_documentation(
    discord_guidelines_urls,
    "discord_guidelines",
    "discord_guidelines_search",
    "Search for Discord guidelines, moderation practices, and community safety information. Use this tool for any questions related to Discord server management and best practices."
)

# Knowledge Management Tool
knowledge_management_urls = [
    "https://research.aimultiple.com/knowledge-management/",
    "https://www.phpkb.com/kb/article/building-a-knowledge-base-with-artificial-intelligence-295.html",
    "https://www.earley.com/insights/knowledge-managements-rebirth-knowledge-engineering-artificial-intelligence",
    "https://support.zendesk.com/hc/en-us/articles/4408831743258",
    "https://www.mojohelpdesk.com/blog/2022/03/ticketing-software-ticketing-system-benefits/",
    "https://www.zendesk.com/blog/ticketing-system-tips/",
    "https://support.zendesk.com/hc/en-us/articles/4408828362522"
]

knowledge_management_tool = load_documentation(
    knowledge_management_urls,
    "knowledge_management",
    "knowledge_management_search",
    "Search for information about knowledge management and ticketing systems. Use this tool for questions related to maintaining knowledge bases, improving system knowledge, tracking questions, and managing ticketing systems. This tool provides insights on AI-powered knowledge management, building knowledge bases, and best practices for ticketing systems."
)

# Community Engagement Tool
community_engagement_urls = [
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4488111/",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4342308/",
    "https://muse.jhu.edu/content/crossref/journals/progress_in_community_health_partnerships_research_education_and_action/v009/9.3.cene.html",
    "https://medium.com/@elle_mccann/building-community-and-engagement-around-data-2fb7d72b13b4",
    "https://ctb.ku.edu/en/table-of-contents/assessment/assessing-community-needs-and-resources/collect-information/main"
]

community_engagement_tool = load_documentation(
    community_engagement_urls,
    "community_engagement",
    "community_engagement_search",
    "Search for information about data collection, reporting, and community engagement initiatives. Use this tool for questions related to monitoring system performance, improving user satisfaction, and fostering community engagement."
)
