import json
import re
from collections import defaultdict

def aggregate_layer_to_block(input_path, output_path):
    # Load input file
    with open(input_path, 'r') as f:
        data = json.load(f)

    layer_accuracy = data['layer_accuracy']

    # Create a default dict to sum scores per block
    block_scores = defaultdict(float)

    # Regular expression to extract block names
    block_pattern = re.compile(r'(layer\d+\.\d+)')

    for layer_name, score in layer_accuracy.items():
        match = block_pattern.search(layer_name)
        if match:
            block_name = match.group(1)
            block_scores[block_name] += score
        else:
            # If it's not part of any block, you can decide to skip or handle it
            # For example, we can ignore conv1 and similar global layers
            print(f"Skipping non-block layer: {layer_name}")

    # Convert defaultdict to regular dict for saving
    block_scores = dict(block_scores)

    # Save output file
    with open(output_path, 'w') as f:
        json.dump(block_scores, f, indent=4)

    print("Aggregation completed successfully.")

# Example usage
input_path = "/home/jose/SkipLess/imprinting/imprinting_ranking.json"
output_path = "aggregated_block_scores.json"
aggregate_layer_to_block(input_path, output_path)
