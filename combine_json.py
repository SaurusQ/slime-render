import os
import json

# Define the folder path that contains the JSON files
folder_path = './configs/'
combined_simconfig = []

# Loop through all the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # Open and read each JSON file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            
            # Append the "simconfig" content to the combined list
            combined_simconfig.extend(data.get('simconfig', []))

# Save the combined "simconfig" into a new JSON file
combined_data = {'simconfig': combined_simconfig}
output_file = os.path.join(folder_path, 'combined_simconfig.json')

with open(output_file, 'w') as outfile:
    json.dump(combined_data, outfile, indent=4)

print(f"Combined simconfig saved to {output_file}")
