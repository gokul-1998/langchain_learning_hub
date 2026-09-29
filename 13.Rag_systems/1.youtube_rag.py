from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv

load_dotenv()


# step1 : Indexing (Document Ingestion)
video_id="QiYKOYLbjVk"

try:
    yt_api=YouTubeTranscriptApi()
    transcript_list=yt_api.fetch(video_id=video_id,languages=['en'])
    # flatten the transcript to plain text

    transcript=" ".join(chunk.text for chunk in transcript_list)
except:
    