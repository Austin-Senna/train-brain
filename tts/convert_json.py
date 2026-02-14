from loader import find_json
import json 
import os 
OUTPUT_PREFIX = "comps_converted"
files = find_json("comps")

OUTPUT_FOLDER = os.path.join(os.getcwd(), OUTPUT_PREFIX)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for file in files:
    output_file = os.path.join(OUTPUT_FOLDER, os.path.basename(file))

    with open(file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
        for line in infile:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # 1. Construct the good and bad sentences
            # Capitalize the first letter so it reads like a proper sentence
            good_sent = f"{data['prefix_acceptable']} {data['property_phrase']}"
            bad_sent = f"{data['prefix_unacceptable']} {data['property_phrase']}"
            
            good_sent = good_sent[0].upper() + good_sent[1:]
            bad_sent = bad_sent[0].upper() + bad_sent[1:]

            # 2. Build the new dictionary matching your target format
            new_data = {
                "sentence_good": good_sent,
                "sentence_bad": bad_sent,
                # "field": "semantics", 
                # "linguistics_term": "concept_property",
                # "UID": data['negative_sample_type'], # Using your negative sample type here
                # "simple_LM_method": True,
                # "one_prefix_method": False,
                # "two_prefix_method": False,
                # "lexically_identical": False,
                # "pairID": str(data['id']),
                # "similarity": data['similarity'] # Keeping this just in case you need it
            }
            
            # 3. Write the new JSON object to the output file
            outfile.write(json.dumps(new_data) + '\n')

        print("Conversion complete!")