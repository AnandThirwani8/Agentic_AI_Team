import os
import glob
from IPython.display import Markdown
from docling.document_converter import DocumentConverter
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document 

from smolagents import Tool
from langchain_core.vectorstores import VectorStore
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, DuckDuckGoSearchTool, VisitWebpageTool

from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings

model = LiteLLMModel(model_id="gemini/gemini-1.5-flash",  api_key=os.environ["GOOGLE_API_KEY"])
# model = LiteLLMModel(model_id="groq/deepseek-r1-distill-llama-70b", api_key=os.environ["GROQ_API_KEY"])
# model = LiteLLMModel(model_id="ollama/deepseek-r1:14b", api_key="")


def create_vector_store(pdf_paths):
    """
    Create a FAISS vector store from multiple PDFs.

    Args:
        pdf_paths (list): List of PDF file paths.

    Returns:
        FAISS vector store
    """

    # PDF Exraction
    pdf_extracted_text = []
    converter = DocumentConverter()
    for pdf in pdf_paths:
        result = converter.convert(pdf)
        markdown_text = result.document.export_to_markdown()
        pdf_extracted_text+=[markdown_text]

    # Convert to documents
    split_documents = [Document(page_content=chunk) for chunk in pdf_extracted_text]

    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(split_documents, embeddings)

    return vectorstore


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(query, k=1)

        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )
    


def create_agentic_team(vectorstore, web_search_required = False):
    """Creates and returns a manager agent that coordinates specialized AI agents."""

    # Create the Analyst Agent for computations and data transformations
    analyst_agent = CodeAgent(
        model=model,
        name="analyst",
        description="Executes complex calculations and data transformations strictly based on retrieved data. It does not generate assumptions, make up information, or infer details beyond the provided input.",
        additional_authorized_imports=['pandas'],
        tools=[]
    )

    # Create the Retriever Agent for document retrieval
    retriever_agent = ToolCallingAgent(
        model=model,
        name="retriever",
        description="Finds and retrieves relevant information from a document database.",
        tools=[RetrieverTool(vectorstore)]
    )

    # Create the Web Researcher Agent for real-time web searches
    web_agent = ToolCallingAgent(
        model=model,
        name="web_researcher",
        description="Performs real-time web searches and retrieves up-to-date information from online sources.",
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()]
    )

    # Create the Manager Agent to coordinate all agents
    if web_search_required:
        AI_team = [analyst_agent, retriever_agent, web_agent]
    else:
        AI_team = [analyst_agent, retriever_agent]

    manager_agent = CodeAgent(
        model=model,
        name = "manager",
        description = "Coordinates and delegates tasks among specialized agents. It returns responses in clear, concise, and human-readable sentences. It does not generate assumptions, make up information, or infer details beyond the provided input",
        managed_agents = AI_team,
        tools=[]
    )

    return manager_agent
