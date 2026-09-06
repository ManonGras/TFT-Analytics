import os
from dotenv import load_dotenv

load_dotenv()

RIOT_API_KEY = os.environ["RIOT_API_KEY"]
REGION_PLATFORM = os.environ.get("RIOT_REGION_PLATFORM", "euw1")
REGION_ROUTING = os.environ.get("RIOT_REGION_ROUTING", "europe")
DATABASE_PATH = os.environ.get("DATABASE_PATH", "data/processed/tft.db")
TFT_SET_NUMBER = int(os.environ.get("TFT_SET_NUMBER", 18))