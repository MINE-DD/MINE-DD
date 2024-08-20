import requests
import json

model = "llama3.1"
template = {
    'article_type': '',
     'study_type': '',
     'title': '',
     'study_scope': {'study_sites': [''], 'description': [''], 'locations': ['']},
     'methodology': [{'name': '',
       'description': '',
       'methods': [''],
       'goals': ['']}],
     'study_duration': {'duration': '', 'details': ['']},
     'participant_age_group': {'age_range': '', 'details': []},
     'weather_data': {'time_period': '',
      'details': [''],
      'measurement': [{'measurement_type': '',
        'description': '',
        'formula': [''],
        'variables': [],
        'notes': ''}]},
     'data_collection': '',
     'authors': '',
     'affiliation': '',
     'citation': '',
     'corresponding_author': '',
     'doi': '',
     'date': '',
     'github_repo': '',
     'journal': ''
}
prompt = (f"You are provided with an excerpt of a text: 'Weather and notified Campylobacter infections in temperate  and sub-tropical regions of Australia  Abstract  Background   The relationship between weather and food-borne diseases has been of great concern  recently. However, the impact of weather variations on food-borne disease may vary in  different areas with various geographic, weather and demographic characteristics. This  study was designed to quantify the relationship between weather variables and  Campylobacter infections in two Australian cities with different local climatic conditions.  Methods   An ecological-epidemiological study was conducted, using weekly disease surveillance  data and meteorological data, over the period 1990-2005, to quantify the relationship  between maximum and minimum temperature, rainfall, relative humidity and  notifications of Campylobacter infections in Adelaide, with a temperate Mediterranean  climate, and Brisbane, with a sub-tropical climate. Spearman correlation and time-series  adjusted Poisson regression analyses were performed taking into account seasonality, lag  effects and long-term trends.   Results   The results indicate that weekly maximum and minimum temperatures were inversely  associated with the weekly number of cases in Adelaide, but positively correlated with  the number of cases in Brisbane, with relevant lagged effects. The effects of rainfall and  relative humidity on Campylobacter infection rates varied in the two cities.' \nYour task is to extract specific information from the text and output it in JSON format according to the following template: {json.dumps(template)}.")

data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100},
}

print(f"Generating a sample user")
response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
json_data = json.loads(response.text)
print(json.dumps(json.loads(json_data["response"]), indent=2))
