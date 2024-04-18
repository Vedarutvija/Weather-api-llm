import locale
locale.getpreferredencoding = lambda: "UTF-8"

import gc, inspect, json, re
import xml.etree.ElementTree as ET
from functools import partial
from typing import get_type_hints
import transformers
import torch
import pickle
from langchain.chains.openai_functions import convert_to_openai_function
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.pydantic_v1 import BaseModel, Field, validator


import requests

class Weather(BaseModel):
    """Provides real time weather information of any location based on specified interest"""
    location: str = Field(description="question of user about a particular location weather details")
    fetched_weather: str = Field(description="answer to provide weather details")

    @validator("location")
    def interests_must_not_be_empty(cls, field):
        if not field:
            raise ValueError("Interest cannot be empty")
        return field

    def fetch_weather(self, api_key):
        """Fetches weather data from OpenWeatherMap API for the specified location"""
        url = f"https://api.openweathermap.org/data/2.5/weather?q={self.location}&appid={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()
                # Process the weather data here (e.g., extract temperature, description)
                self.fetched_weather = f"Weather in {data['name']}: {data['weather'][0]['description']}"  # Example processing
            except Exception as e:
                print(f"Error processing weather data: {e}")
                self.fetched_weather = f"Error: Could not process weather data for {self.location}"
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            self.fetched_weather = f"Error: Could not fetch weather data for {self.location}"

# Example usage (replace 'YOUR_API_KEY' with your actual OpenWeatherMap API key)
weather = Weather(location="Hyderabad", fetched_weather="Weather data not fetched yet")
weather.fetch_weather('438a6e81425db473da06b9ed05071cea')
print(weather.fetched_weather)
