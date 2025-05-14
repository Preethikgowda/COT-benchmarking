
import json
import re
from groq import Groq
from openai import OpenAI
from .prompts import cot_prompt_ent, cot_prompt_rel
from .utils import *
import os
#from dotenv import load_dotenv
#oad_dotenv() 

class EntityExtractionCOT:
    _valid_combinations = {
        "groq": ["Gemma-7b-It","deepseek-r1-distill-qwen-32b", "llama-3.2-3b-preview", "llama3-70b-8192", "Gemma2-9b-It",
             "deepseek-r1-distill-llama-70b", "qwen-2.5-32b", "qwen-2.5-coder-32b"],
        "openai": ["gpt-4o-mini"],
        "gemini":["gemini-2.0-flash","gemini-1.5-pro"]
    }
    
    def __init__(self,service_config:dict=None):
        '''
        for groq client, the model should be one of the following: llama3-70b-8192, llama3-8b-8192, Gemma2-9b-It
        To initialize the client, pass the service_config as a dictionary with the following
        service_config={'client':'groq','model':'llama3-70b-8192','api_key':'api_key','iteration':3}        
        
        for openai client, the model should be one of the following: gpt-4o-mini
        To initialize the client, pass the service_config as a dictionary with the following
        service_config={'client':'openai','model':'gpt-4o-mini','api_key','iteration':3}
        
        
        
        '''
        if service_config is None:
            print("No configuration provided using default config")
            service_config = {'client':'openai','model':'gpt-4o-mini','iteration':3}
            
        if service_config['client']=='groq':
            self.model = service_config.get('model', "Gemma2-9b-It")
            self.api=service_config.get('api_key','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')
            self.client=Groq(api_key=self.api)
            print(f"Using client: {service_config['client']}")
            print(f"Using model: {self.model}")
            if service_config['model'] not in self._valid_combinations['groq']:
                raise ValueError(f"Model '{service_config['model']}' is not valid for client 'groq'")
            
        elif service_config['client']=='openai':

            self.api_key = os.getenv("OPENAI_API_KEY")
            self.model = service_config.get('model','gpt-4o-mini')
            self.client=AzureOpenAIClient(api_key=api_key, deployment=self.model)
            print(f"Using client: {service_config['client']}")
            print(f"Using model: {self.model}")
            if service_config['model'] not in self._valid_combinations['openai']:
                raise ValueError(f"Model '{service_config['model']}' is not valid for client 'openai'")
        
        elif service_config['client'] == 'gemini':
            self.api_key = os.getenv("GEMINI_API_KEY")
            self.model = service_config.get('model', 'gemini-2.0-flash')
            genai.configure(api_key=api_key) 
            self.client = genai.GenerativeModel(self.model) 
            print(f"Using client: {service_config['client']}")
            print(f"Using model: {self.model}")
        
        else:
            print('please select the valid client')
        
        self.iteration=service_config['iteration']
        self.service_config=service_config
        
    def string_ent(self,entities, paragraph):
        
        system_message=cot_prompt_ent(entities,paragraph)[0]
        user_message=cot_prompt_ent(entities,paragraph)[1]

        if self.service_config['client']=='openai':
            try:
                chat_completion = self.client.send_message(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                )
                response=chat_completion['choices'][0]['message']['content']
                # print(f"the questions geqnaated are {response}")
                return response
            except Exception as e:
                print(f"Error occurred for model: {e}")
                return None
        elif self.service_config['client'] == 'groq':
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model,
                    temperature=0.1
                )
                response=chat_completion.choices[0].message.content
                # print(f"the questions geqnaated are {response}")
                return response
            except Exception as e:
                print(f"Error occurred for model: {e}")
                return None
        elif self.service_config['client'] == 'gemini':
            try:
                chat_session = self.client.start_chat()
                chat_response = chat_session.send_message(
                    f"System: {system_message}\nUser: {user_message}"
                )
                return chat_response.text
            except Exception as e:
                print(f"Error occurred for Gemini model: {e}")
                return None
        
        else:
            print("Invalid client specified.")
            return None



    def process_ent(self, paragraph: str, initial_entities: list = None):
        paragraph = paragraph.replace("'", "`")
        paragraph = paragraph.replace('"', "`")
        if initial_entities is None:
            initial_entities=[]
        initial_entities = list(initial_entities)
        entities = initial_entities.copy()
        for iter in range(self.iteration):
            text = self.string_ent(entities, paragraph)
            if text:
                match = re.search(r'list_of_new_entities\s*=\s*(\[[^\]]*\])', text)
                if match:
                    try:
                        new_entities = eval(match.group(1))
                        entities.extend(new_entities)
                        entities = list(set(entities))
                        print(f'Iteration :{ iter + 1}')
                        print("New Entities :", entities)
                    except SyntaxError:
                        print("syntax if invalid")
                else:
                    print("List not found.")
            else:
                print("No response for model")
                
        entities=list(set(entities))
        ent = {'entities': entities}
        with open(r'entity_cot\json_folder\entity.json', 'w') as json_file:
            json.dump(ent, json_file,indent=4)
            print('Entities have been saved as tuples')
        return entities 



    def string_rel(self,paragraph,entities):
        system_message=cot_prompt_rel(entities, paragraph)[0]
        user_message=cot_prompt_rel(entities, paragraph)[1]


        if self.service_config['client']=='openai':
            try:
                chat_completion = self.client.send_message(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                )
                response=chat_completion['choices'][0]['message']['content']
                # print(f"the questions geqnaated are {response}")
                return response.replace("\n"," ").replace("`","")
            except Exception as e:
                print(f"Error occurred for model: {e}")
                return None
        elif self.service_config['client'] == 'groq':
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model,
                    temperature=0.1
                )
                response = chat_completion.choices[0].message.content
                return response
            except Exception as e:
                print(f"Error occurred for Groq model: {e}")
                return None
    
        elif self.service_config['client'] == 'gemini':
            try:
                chat_session = self.client.start_chat()
                chat_response = chat_session.send_message(
                    f"System: {system_message}\nUser: {user_message}"
                )
                return chat_response.text
            except Exception as e:
                print(f"Error occurred for Gemini model: {e}")
                return None
    
        else:
            print("Invalid client specified.")
            return None

    def process_rel(self, paragraph, entities):
        paragraph = paragraph.replace("'", "`")
        paragraph = paragraph.replace('"', "`")

        text = self.string_rel(paragraph, entities) 
        
        pattern = r"list_of_triplets\s*=\s*\[(.*?)\]"
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            print("No relationships found in text")
            return []

        try:
            relationships_str = match.group(1).strip()
            relationships_str = re.sub(r"#.*?\n", "", relationships_str)
            relationships_str = re.sub(r"//.*?\n", "", relationships_str)
            
            relationships_str = relationships_str.replace("(", "[").replace(")", "]")
            relationships_str = f"[{relationships_str}]"
            
            relationships = json.loads(relationships_str)
            if isinstance(relationships, list):
                return [tuple(rel) if isinstance(rel, list) else rel for rel in relationships]
            return []
            
        except Exception as e:
            print(f"Error parsing relationships: {e}")
            return []