def cot_prompt_ent(entities, paragraph):
    """
    Generates a structured CoT (Chain of Thought) prompt for extracting entities from a paragraph.

    Args:
    entities: List of entities already provided for reference.
    paragraph: The text from which new entities need to be extracted.

    Returns:
    A tuple containing the system and user messages.
    """
    
    system_message = f'''
    You are a highly experienced assistant specialized in Knowledge Graphs and entity extraction. Your task is to extract meaningful and unique entities 
    from the provided paragraph using a clear step-by-step approach. 

    **Definitions**:
    - `Entities`: An entity is a distinct, meaningful, and contextually significant element or concept within a given text. Entities typically represent real-world objects, concepts, or ideas that can be categorized and identified for tasks such as knowledge graph construction, semantic analysis, or information extraction, terms such as Person, Organization, Location, or other significant elements.

    **Instructions**:
    1. Compare the provided list of entities: {entities} with the paragraph content.
    2. Identify unique entities from the paragraph that do not appear in the given list of entities.
    3. Ensure entities are factually accurate and formatted exactly as they appear in the paragraph (maintain case, structure, spelling, etc.).
    4. Exclude non-entities (stop words, irrelevant words, or generic terms like "branches").
    5. If no new entities are found, return `list_of_new_entities = []`.

    **Output**:
    - Always provide the extracted entities as a list under the variable `list_of_new_entities`. Use double quotes (`"`) for all entities in the list and if you want to highlight or mention something, then use backticks (') to highlight or mention it.
    - Present the list only after explaining the step-by-step process.
    - Do not include duplicate or irrelevant entities.
    - Do not add entities already mentioned in the given list.
    
    **Example**:
    Given the sentence:
   `"Marie Curie was the first woman to win a Nobel Prize and conducted research on radioactivity in 1983."`

    Correct Extraction:
    ```python
    list_of_new_entities = ["Marie Curie", "Nobel Prize", "radioactivity", "1903"]

    Incorrect Extraction:
    ```python
    list_of_new_entities = ["first woman to win a Nobel Prize", "conducted research", "in 1903"]
    ```
    
    '''
    
    user_message = f'''
    Below is your task for extracting new entities:
    
    **Input**:
    list_of_entities = {entities}
    paragraph = "{paragraph}"
    
    **Expected Step-by-Step Approach**:
    1. Carefully review the `list_of_entities` and compare it against the paragraph.
    2. Identify all potential entities within the paragraph. This includes proper nouns, specific terms, and unique phrases not present in the input list.
    3. Verify each entity’s accuracy and ensure it aligns with the paragraph context.
    4. Entities should act as in importance in the paragram.
    4. Exclude duplicates and entities already listed in the input.
    5. Provide the final result as `list_of_new_entities`. Format the response as follows:

    **Output Example**:
    ```
    First, I compared the provided `list_of_entities` with the paragraph's content.
    Then, I identified unique terms such as [example terms].
    Entities that are already in the input list, like [example terms], were excluded.
    The remaining valid entities were added to the new list.
    Final Output:
    list_of_new_entities = ["entity_1", "entity_2", "entity_3"]
    ```
    - If no new entities are found, your response must be: `list_of_new_entities = []`, but you must explain why.
    - If `list_of_entities` is empty, fill it with newer extracted list step-by-step 
    - Most important, If paragraph contains any numeraical pattern could be pricing, phone no, id, etc consider this numerical pattern in entities. 
    '''
    
    return system_message, user_message

def cot_prompt_rel(entities, paragraph):
    """
    Generates a structured CoT (Chain of Thought) prompt for extracting relationships between entities in a paragraph.

    Args:
    entities: List of entities already provided for reference.
    paragraph: The text from which relationships need to be extracted.

    Returns:
    A tuple containing the system and user messages.
    """
    
    system_message = f"""
    You are a skilled assistant specializing in Knowledge Graphs and relationship extraction between entities. 
    Your task is to identify meaningful and factually accurate relationships between the provided entities from the paragraph using a step-by-step approach.

    **Definitions**:
    - `Entities`: These are real-world objects, people, places, or concepts provided in the list: {entities}.
    - `Relationships`: A factual and meaningfull connection or interaction between two entities explicitly stated in the paragraph, and the realtionship shold be in uppercase only  
      such as "OWNS", "WORKS_AT", "FRIENDS_WITH", or "LOCATED_IN".

    **Instructions**:
    1. Compare the list of entities {entities} with the paragraph content.
    2. Identify explicit relationships that connect the entities mentioned in the paragraph.
    3. For each relationship, construct a triplet as a Python-style tuple in the format:
        ("<Entity 1>", "<Relationship>", "<Entity 2>")

    4. Do not include any inferred or guessed relationships. Use only those explicitly mentioned in the paragraph.
    5. If no relationships are found, return an empty list as `list_of_triplets=[]`.
    6. Consider those only entities from the list of entities as the way they there (cases, words) are in `source_node` and `target_node`.
    **Output**:
    - Present the relationships only after explaining the step-by-step process of identification.
    - Provide the final output as list_of_triplets, which is a list of tuples, each representing a relationship.
    
    
    Note - output must contain `list_of_triplets` = [("<Entity 1>", "<Relationship>", "<Entity 2>")]

    """
    
    {
    "examples": [
        {
            "text": "Elon Musk founded SpaceX in 2002 to revolutionize space travel.",
            "correct_relations": [
            ["Elon Musk", "CEO", "SpaceX"],
            ["SpaceX", "FOUNDED", "Elon Musk"]
            ],
            "incorrect_relations": [
                ["Elon Musk", "founded", "SpaceX", "2002"],
                ["Elon Musk", "established", "SpaceX", "in 2002"]
            ]
        },
        {
            "text": "Microsoft partnered with OpenAI in 2019 to develop advanced AI models.",
            "correct_relations": [
                ["Microsoft", "OWNS", "OpenAI"],
                ["OpenAI", "ACQUIRED", "Microsoft"]
            ],
            "incorrect_relations": [
                ["Microsoft", "partnered with", "OpenAI", "2019"],
                ["Microsoft", "collaborated with", "OpenAI", "2019"]
            ]
        }
    ]
}

    
  
    user_message = f"""
    Below is your task for extracting relationships between entities:

    **Input**:
    list_of_entities = {entities}
    paragraph = "{paragraph}"
    
    **Expected Step-by-Step Approach**:
    1. Compare the `list_of_entities` with the content of the `paragraph`.
    2. Identify all pairs of entities in the `paragraph` that have a factual connection or interaction.
    3. For each identified pair, determine the specific relationship mentioned in the `paragraph`.
    4. Refer `paragraph` only to find the relationship or patther from `list_of_entities`
    4. Format each relationship as a tuple containing three values: (source_node, relationship, target_node), in that exact order
    5. Return the relationships as `list_of_triplets=[...]`not just 'list_of_triplets'.

    **Output Example**:
    ```
    First, I compared the `list_of_entities` with the paragraph's content.
    Then, I identified connections such as:
      - "sam" is working at "Bajaj org.", which establishes a WORKING relationship.
      - "Vikram Baji" is a friend of "Samuel Oak", which establishes a FRIEND relationship.
    The final result is:
    list_of_triplets = [
    ("sam", "WORKING", "Bajaj org."),
    ("Vikram Baji", "FRIEND", "Samuel Oak")
]

    ```
    - If no relationships are found, your response must be: `list_of_triplets = []`, but you must explain why.

    **Guidelines**:
    1. Ensure factual accuracy—extract relationships strictly based on the paragraph.
    2. Avoid inferred, implied, or guessed relationships.
    3. Use Python tuple formatting: wrap each triplet in parentheses and separate values by commas. Strings should be enclosed in double quotes.
    """
    
    return system_message, user_message