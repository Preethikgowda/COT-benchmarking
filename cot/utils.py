import json
import re

default_config={'client':'groq','model':'Gemma-7b-It'}

def is_json(data):
    try:
        json.loads(data)
        return True
    except (ValueError, TypeError):
        return False

def bypass_chunk(ee, paragraph, metadata: dict = None, output_json: str = None):
    paragraph = paragraph.replace("'", "`")
    paragraph = paragraph.replace('"', "`")

    entities = ee.process_ent(paragraph)  
    ner = ee.process_rel(paragraph, entities)  

    if not ner:
        return entities, []  

    to_store = []
    for triplet in ner:
        if len(triplet) == 3:  
            if metadata:
                metadata['page_content'] = paragraph
                triplet = list(triplet) + [metadata]
            to_store.append(triplet)
    
    if output_json:
        try:
            with open(output_json, 'w', encoding='utf-8') as file:
                json.dump(to_store, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving JSON: {e}")

    return entities, ner

def empty_structure():
    data=[
    {
        "source_node": " ",
        "relationship": " ",
        "target_node": " "
        
    }]
    return data



import requests

class AzureOpenAIClient:
    def __init__(self, api_key, deployment, api_version='2024-08-01-preview'):
        self.api_key = api_key
        self.api_version = api_version
        self.base_url = f'https://openai-chatbots.openai.azure.com/openai/deployments/{deployment}/chat/completions'
    
    def send_message(self, messages):
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.api_key
        }
        data = {"messages": messages}

        response = requests.post(f'{self.base_url}?api-version={self.api_version}', headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()