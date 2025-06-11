import os
import csv
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_csv_to_json(csv_file_path, output_file_path=None):
    """
    Convert ICD-10 CSV data to a hierarchical JSON structure based on level information.
    
    Args:
        csv_file_path (str): Path to the CSV file
        output_file_path (str, optional): Path to save the JSON output. If None, will use the same name as CSV but with .json extension.
    
    Returns:
        str: Path to the created JSON file
    """
    try:
        # Handle output file path
        if output_file_path is None:
            output_file_path = Path(csv_file_path).with_suffix('.json')
        
        logger.info(f"Converting {csv_file_path} to JSON")
        
        # Read CSV file with proper handling of quotes and special characters
        data = []
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
                # Try to detect the delimiter
                sample = csv_file.read(1024)
                csv_file.seek(0)
                
                if ',' in sample:
                    delimiter = ','
                elif ';' in sample:
                    delimiter = ';'
                elif '\t' in sample:
                    delimiter = '\t'
                else:
                    delimiter = ','
                
                csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
                for row in csv_reader:
                    data.append(row)
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            with open(csv_file_path, 'r', encoding='latin-1') as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
                for row in csv_reader:
                    data.append(row)
        
        if not data:
            logger.error("No data found in the CSV file")
            return None
        
        # Identify the level column (could be 'Level', 'level', 'LEVEL', etc.)
        level_column = None
        for column in data[0].keys():
            if 'level' in column.lower():
                level_column = column
                break
        
        if not level_column:
            logger.warning("No level column found. Treating all entries as same level.")
            # Create a flat structure if no level information is available
            result = {"icd_codes": data}
        else:
            # Create hierarchical structure based on levels
            result = build_hierarchical_structure(data, level_column)
        
        # Write to JSON file
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully converted to JSON. Output saved to {output_file_path}")
        return output_file_path
    
    except Exception as e:
        logger.error(f"Error converting CSV to JSON: {e}")
        return None

def build_hierarchical_structure(data, level_column):
    """
    Build a hierarchical structure from flat data based on level information.
    
    Args:
        data (list): List of dictionaries containing the CSV data
        level_column (str): Name of the column containing level information
    
    Returns:
        dict: Hierarchical structure of the data
    """
    # Sort data by level to ensure parent nodes come before children
    try:
        # Try to convert level to integer if possible
        for item in data:
            try:
                item[level_column] = int(item[level_column])
            except (ValueError, TypeError):
                pass
        
        sorted_data = sorted(data, key=lambda x: x[level_column])
    except Exception as e:
        logger.warning(f"Could not sort by level: {e}. Using original order.")
        sorted_data = data
    
    # Create root structure
    result = {"icd_codes": []}
    
    # Stack to keep track of parent nodes at each level
    node_stack = [result["icd_codes"]]
    current_level = None
    
    for item in sorted_data:
        item_level = item.get(level_column)
        
        # Create a copy of the item without the level column for cleaner output
        node_data = {k: v for k, v in item.items() if k != level_column}
        
        # Initialize children array
        node_data["children"] = []
        
        if current_level is None:
            # First item
            current_level = item_level
            node_stack[0].append(node_data)
        elif item_level > current_level:
            # Child node - go deeper in the hierarchy
            parent = node_stack[-1][-1]
            node_stack.append(parent["children"])
            node_stack[-1].append(node_data)
            current_level = item_level
        elif item_level < current_level:
            # Go back up in the hierarchy
            while len(node_stack) > 1 and item_level <= current_level:
                node_stack.pop()
                current_level -= 1
            node_stack[-1].append(node_data)
            current_level = item_level
        else:
            # Same level as previous item
            node_stack[-1].append(node_data)
    
    # Clean up empty children arrays
    def remove_empty_children(node):
        if isinstance(node, dict) and "children" in node:
            if not node["children"]:
                del node["children"]
            else:
                for child in node["children"]:
                    remove_empty_children(child)
        elif isinstance(node, list):
            for item in node:
                remove_empty_children(item)
    
    remove_empty_children(result)
    return result

def main():
    # Path to the CSV file
    kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KB")
    csv_file_path = os.path.join(kb_dir, "ICD 10 Additional Data.csv")
    
    # Convert CSV to JSON
    output_file_path = os.path.join(kb_dir, "icd10_additional_data.json")
    result = convert_csv_to_json(csv_file_path, output_file_path)
    
    if result:
        logger.info(f"Conversion completed successfully. JSON file saved at: {result}")
    else:
        logger.error("Conversion failed.")

if __name__ == "__main__":
    main()