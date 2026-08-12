import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result

from config import RIOT_API_KEY, REGION_PLATFORM, REGION_ROUTING


def _is_rate_limited(response):
    return response is not None and response.status_code == 429


@retry(
    retry=retry_if_result(_is_rate_limited),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
)
def _get(url, params=None):
    params = params or {}
    params["api_key"] = RIOT_API_KEY
    response = requests.get(url, params=params, timeout=10)
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 5))
        time.sleep(retry_after)
        return response
    response.raise_for_status()  # raises for 4xx/5xx (except 429 handled above)
    return response


def get_challenger_league():
    url = f"https://{REGION_PLATFORM}.api.riotgames.com/tft/league/v1/challenger"
    return _get(url).json()


def get_grandmaster_league():
    url = f"https://{REGION_PLATFORM}.api.riotgames.com/tft/league/v1/grandmaster"
    return _get(url).json()


def get_summoner_by_id(summoner_id):
    """Kept for completeness: less used in the main flow since
    puuid is provided directly by ladder entries."""
    url = f"https://{REGION_PLATFORM}.api.riotgames.com/tft/summoner/v1/summoners/{summoner_id}"
    return _get(url).json()


def get_match_ids_by_puuid(puuid, count=20):
    url = f"https://{REGION_ROUTING}.api.riotgames.com/tft/match/v1/matches/by-puuid/{puuid}/ids"
    return _get(url, params={"count": count}).json()


def get_match_by_id(match_id):
    url = f"https://{REGION_ROUTING}.api.riotgames.com/tft/match/v1/matches/{match_id}"
    return _get(url).json()