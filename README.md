import os
import re

def parse_feature_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    feature_match = re.search(r'Feature: (.*)', content)
    feature_name = feature_match.group(1).strip() if feature_match else None

    scenario_blocks = re.split(r'Scenario:', content)[1:]
    scenarios = []

    for block in scenario_blocks:
        lines = block.strip().splitlines()
        scenario_name = lines[0].strip()
        steps = []
        temp_step = None
        temp_data = []

        for line in lines[1:]:
            line = line.strip()
            if line.startswith(("Given", "When", "Then", "And")):
                if temp_step and temp_data:
                    steps.extend(substitute_step(temp_step, temp_data))
                    temp_step = None
                    temp_data = []
                temp_step = line
            elif line.startswith("|"):
                temp_data.append([item.strip() for item in line.strip("|").split("|")])
            else:
                if temp_step and not temp_data:
                    steps.append(temp_step)
                    temp_step = None

        if temp_step and temp_data:
            steps.extend(substitute_step(temp_step, temp_data))
        elif temp_step:
            steps.append(temp_step)

        scenarios.append({"name": scenario_name, "steps": steps})

    return feature_name, scenarios

def substitute_step(step, data):
    headers = data[0]
    rows = data[1:]
    steps = []

    for row in rows:
        step_text = step
        data_dict = dict(zip(headers, row))
        for key, value in data_dict.items():
            step_text = re.sub(rf'"{key}"|\<{key}\>', value, step_text)
        steps.append(step_text)

    return steps

def read_feature_files(folder_path):
    feature_files = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".feature"):
            file_path = os.path.join(folder_path, file_name)
            feature_name, scenarios = parse_feature_file(file_path)
            feature_files.append((file_name, feature_name, scenarios))

    return feature_files

folder_path = 'features'
feature_files = read_feature_files(folder_path)

for file_name, feature_name, scenarios in feature_files:
    print(f"File: {file_name}")
    print(f"Feature: {feature_name}")
    for scenario in scenarios:
        print(f" Scenario: {scenario['name']}")
        for step in scenario["steps"]:
            print(f" {step}")
    print("------------------------")
