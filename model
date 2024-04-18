import locale
locale.getpreferredencoding = lambda: "UTF-8"
import os
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
model_name = "nilq/mistral-1L-tiny"


def load_model(model_name: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    with torch.device("cpu:0"):
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()

    return tokenizer, model
     
tokenizer, model = load_model(model_name)

class Weather(BaseModel):
    location: str = Field(description="question of user about a particular location weather details")
    fetched_weather: str = Field(default="", description="answer to provide weather details")

    @validator("location")
    def interests_must_not_be_empty(cls, field):
        if not field:
            raise ValueError("Interest cannot be empty")
        return field

    def fetch_weather(self, api_key):
        url = f"https://api.openweathermap.org/data/2.5/weather?q={self.location}&appid={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()
                # Process the weather data here (e.g., extract temperature, description)
                weather_description = data['weather'][0]['description']
                rain = 'no rain'  # Assume no rain by default
                if 'rain' in weather_description.lower():
                    rain = 'rain'
                self.fetched_weather = f"Weather in {data['name']}: {weather_description}. There is a chance of {rain} today."
            except Exception as e:
                print(f"Error processing weather data: {e}")
                self.fetched_weather = f"Error: Could not process weather data for {self.location}"
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            self.fetched_weather = f"Error: Could not fetch weather data for {self.location}"


def generate_response(prompt):
  # Use the LLModel to generate a response based on the prompt
  input_ids = tokenizer.encode(prompt, return_tensors="pt")
  output = model.generate(input_ids, max_length=50, do_sample=True)
  response = tokenizer.decode(output[0], skip_special_tokens=True)

  # Fetch weather information if the prompt asks about weather
  if "weather" in prompt.lower():
      # Extract the location from the prompt (assuming it's mentioned after "weather in")
      location_start = prompt.lower().find("weather in") + len("weather in ")
      location_end = prompt.lower().find("?", location_start)
      if location_start > -1 and location_end > -1:
          location = prompt[location_start:location_end].strip()
          weather = Weather(location=location)
          weather.fetch_weather('438a6e81425db473da06b9ed05071cea')  # add API key
          response = response + "\n" + weather.fetched_weather
      else:
          response = response + "\nI couldn't identify a location in your prompt. Please specify the location (e.g., 'What is the weather in London today?') "
  return response


prompt = str(input("Ask questions about weather conditions of a city"))
response = generate_response(prompt)
print(response)

import pickle

# Serialize tokenizer and model objects
tokenizer_bytes = pickle.dumps(tokenizer)
model_bytes = pickle.dumps(model)

# Save serialized objects to a file
with open("model.pkl", "wb") as f:
    f.write(tokenizer_bytes)
    f.write(model_bytes)

print("Model saved successfully.")
     
