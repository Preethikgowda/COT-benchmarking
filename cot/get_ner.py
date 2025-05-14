import json
from cot.chain_of_thought import *

# Load dataset
def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

# Process a paragraph or chunks using Chain of Thought
def cot_test(paragraph: str = None, chunks: list = None, metadata: dict = None, output_json: str = None, service_config: dict = None):
    if paragraph and chunks:
        raise ValueError("Please provide either paragraph or chunks.")
    
    if service_config:
        ee = EntityExtractionCOT(service_config)
    else:
        ee = EntityExtractionCOT()

    all_entities = []
    all_relationships = []

    if chunks:
        for chunk in chunks:
            entities, relationships = bypass_chunk(ee, chunk, metadata, output_json)
            all_entities.extend(entities)
            all_relationships.extend(relationships)
    elif paragraph:
        entities, relationships = bypass_chunk(ee, paragraph, metadata, output_json)
        all_entities.extend(entities)
        all_relationships.extend(relationships)
    else:
        return None, None

    return all_entities, all_relationships

# Process an individual paragraph and extract entities & relationships
def bypass_chunk(ee, paragraph, metadata: dict = None, output_json: str = None):
    paragraph = paragraph.replace("'", "`")
    paragraph = paragraph.replace('"', "`")

    entities = ee.process_ent(paragraph)  # Extract entities
    ner = ee.process_rel(paragraph, entities)  # Extract relationships

    if not ner:
        return entities, []  

    to_store = []
    for triplet in ner:
        if metadata:
            metadata['page_content'] = paragraph
            triplet.append(metadata)  
        to_store.append(triplet)
    processed = str(to_store).replace("'", '"')

    if output_json:
        try:
            with open(output_json, 'w', encoding='utf-8') as file:
                json.dump(to_store, file, indent=4)
        except Exception as e:
            print(f"Error saving JSON: {e}")

    return entities, ner

# Process dataset from JSON file and extract entities & relationships
def process_dataset(dataset, output_file="extracted_results.json"):
    results = {}

    service_config = {'client': 'groq', 'model': 'deepseek-r1-distill-llama-70b', 'iteration': 3}
    ee = EntityExtractionCOT(service_config)

    for text_id, text_data in dataset.items():
        for paragraph, expected_entities in text_data.items():
            print(f"\nProcessing: {text_id}...")

            # Extract entities & relationships
            extracted_entities, extracted_relationships = cot_test(paragraph=paragraph, service_config=service_config)

            results[text_id] = {
                "paragraph": paragraph,
                "extracted_entities": extracted_entities,
                "extracted_relationships": extracted_relationships
            }

            print(f"Extracted Entities: {extracted_entities}")
            print(f"Extracted Relationships: {extracted_relationships}")

    # Save results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n Extracted entities and relationships saved to {output_file}")

# Run the extraction process
if __name__ == "__main__":
    dataset_path = "data2.json"  
    dataset = load_dataset(dataset_path)
    process_dataset(dataset)

