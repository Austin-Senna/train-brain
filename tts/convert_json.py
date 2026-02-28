import json
import os

# Define input and output paths
input_file = "/projects/bgbh/awijaya/train-brain/tts/winogrande_1.1/train_l.jsonl"
output_file = "winogrande_converted.jsonl"

# Ensure the directory exists if you specify a folder path for the output
os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        if not line.strip():
            continue
            
        data = json.loads(line)
        
        sentence = data["sentence"]
        opt1 = data["option1"]
        opt2 = data["option2"]
        ans = data["answer"]
        
        # 1. Determine which option is correct based on the "answer" key
        if ans == "1":
            good_word = opt1
            bad_word = opt2
        elif ans == "2":
            good_word = opt2
            bad_word = opt1
        else:
            # Fallback just in case there's unexpected data
            continue
            
        # 2. Construct the good and bad sentences by replacing the blank '_'
        good_sent = sentence.replace("_", good_word)
        bad_sent = sentence.replace("_", bad_word)

        # 3. Build the new dictionary
        new_data = {
            "sentence_good": good_sent,
            "sentence_bad": bad_sent,
        }
        
        # 4. Write the new JSON object to the output file
        outfile.write(json.dumps(new_data) + '\n')

print(f"Conversion complete! Output saved to {output_file}")