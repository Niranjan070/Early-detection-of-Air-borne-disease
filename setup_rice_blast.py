
import os

# Update spore_classes.yaml
with open('configs/spore_classes.yaml', 'w') as f:
    f.write('''# Spore Classification Classes
# Define all spore types to be detected

# Number of classes
nc: 1

# Class names
names:
  0: magnaporthe_oryzae

# Class descriptions
descriptions:
  magnaporthe_oryzae: "Pear-shaped (pyriform), usually 3-celled spores causing Rice Blast"

# Detection difficulty (for reference)
difficulty:
  magnaporthe_oryzae: "medium"
''')

# Update disease_mapping.yaml
with open('configs/disease_mapping.yaml', 'w') as f:
    f.write('''# Spore Type to Plant Disease Mapping
# This file maps detected spore types to potential plant diseases

disease_mapping:
  magnaporthe_oryzae:
    diseases:
      - name: "Rice Blast"
        crops: ["rice", "wheat", "barley"]
        severity: "critical"
    threshold_low: 5
    threshold_high: 20
''')

print("Config files updated successfully.")
