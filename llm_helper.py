from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool

from pypdf import PdfReader
from pinecone_helper import *

def read_file(file):
    content = ""
    # Create a pdf reader object
    reader = PdfReader(file)

    for page in reader.pages:
        # For each page in the pdf add it to the content string
        content += page.extract_text()

    # Create a recursive text splitter object
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."],  # Characters on which to split the text
        chunk_size = 200, # Number of words in a chunk
        chunk_overlap = 50  # Numbers of words which will be overlapped with the previous chunk
    )

    chunks = splitter.split_text(content)

    create_vector_db()
    insert_into_vector_db(chunks)

def query_tool(text):
    """
    RAG Tool to retrieve information from the indexed PDF in Pinecone.
    This tool does NOT access any local files. It searches within the
    PDF content that has already been embedded and stored.
    """

    # Carry out the user's query on the database
    context = query_vector_db(text)
    return context


def create_agent():


    # Create an llm object using Google class and gemini 2.5 flash model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


    # Create a tool object using Tavily Search
    search_tool = TavilySearch(
        max_results = 3,
        topic = "general",
        
    )

    rag_tool = Tool(
        name="rag",
        func=query_tool,
        description="Use this tool to answer questions **based on the PDF content** that has already been embedded and stored in Pinecone. The user may refer to it as 'the file'."
    )

    # Create an AI agent by merging the llm with the tool
    agent = create_react_agent(llm, [search_tool, rag_tool])

    return agent

def get_llm_response(agent, user_input):
    # Make a call to the llm and return the response
    resp = agent.invoke({"messages": user_input})

    try:
        return resp["messages"][3].content
    
    except:
        return resp["messages"][1].content


if __name__ == "__main__":
    agent = create_agent()
    print(get_llm_response(agent,"Which countries are hosting the 2026 football world cup"))
