import os
import json
import argparse

parser = argparse.ArgumentParser(description="Combine simconfig JSON files and optionally increment frame numbers.")
parser.add_argument("folder_path", type=str, help="Folder path that contains the JSON files")
parser.add_argument("-i", "--increment-frame", action="store_true", help="Increment the frame numbers in the combined output")
args = parser.parse_args()

folder_path = args.folder_path
combined_simconfig = []
frame_counter = 0
files = sorted(os.listdir(folder_path))

for filename in files:
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            
            # Iterate through the simconfig items and update the frame if needed
            for config in data.get('simconfig', []):
                if args.increment_frame:
                    config['frame'] = frame_counter
                    frame_counter += 1
                combined_simconfig.append(config)

combined_data = {'simconfig': combined_simconfig}
output_file = 'combined_simconfig.json'

with open(output_file, 'w') as outfile:
    json.dump(combined_data, outfile, indent=4)

print(f"Combined simconfig saved to {output_file}")
