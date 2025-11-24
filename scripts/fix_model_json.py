import json
import os

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../public/model')
MODEL_JSON_PATH = os.path.join(MODEL_DIR, 'model.json')

def convert_inbound_node_keras3_to_keras2(node):
    """
    Convert Keras 3 inbound node format to Keras 2/TF.js format.
    
    Keras 3 format:
    {
        "args": [tensor_spec, ...],
        "kwargs": {...}
    }
    
    Keras 2/TF.js format:
    [tensor_spec, 0, 0, kwargs]
    """
    if isinstance(node, dict) and 'args' in node:
        # Extract args - this is the list of input tensors
        args = node.get('args', [])
        kwargs = node.get('kwargs', {})
        
        # In Keras 2 format, each inbound node is:
        # [input_tensor, node_index, tensor_index, kwargs]
        # For simplicity, we'll convert the first arg and add indices
        if args:
            # Return the Keras 2 format
            # [layer_name, node_index, tensor_index, kwargs]
            return [args[0], 0, 0, kwargs] if kwargs else [args[0], 0, 0]
        return [None, 0, 0, kwargs] if kwargs else []
    
    return node

def fix_inbound_nodes_recursive(obj, path=""):
    """Recursively fix inbound_nodes structure"""
    if isinstance(obj, dict):
        # Fix inbound_nodes if present
        if 'inbound_nodes' in obj:
            inbound_nodes = obj['inbound_nodes']
            if isinstance(inbound_nodes, list) and len(inbound_nodes) > 0:
                # Check if this is Keras 3 format (objects with 'args' and 'kwargs')
                first_node = inbound_nodes[0]
                if isinstance(first_node, dict) and 'args' in first_node:
                    print(f"Converting inbound_nodes at {path}")
                    obj['inbound_nodes'] = [
                        convert_inbound_node_keras3_to_keras2(node) 
                        for node in inbound_nodes
                    ]
        
        # Recursively process all values
        for key, value in list(obj.items()):
            new_path = f"{path}.{key}" if path else key
            obj[key] = fix_inbound_nodes_recursive(value, new_path)
    
    elif isinstance(obj, list):
        # Recursively process all list items
        return [fix_inbound_nodes_recursive(item, f"{path}[{i}]") 
                for i, item in enumerate(obj)]
    
    return obj

def fix_batch_shape_recursive(obj):
    """Recursively fix batch_shape to batchInputShape"""
    if isinstance(obj, dict):
        if 'batch_shape' in obj:
            obj['batchInputShape'] = obj.pop('batch_shape')
        
        for key, value in list(obj.items()):
            obj[key] = fix_batch_shape_recursive(value)
    
    elif isinstance(obj, list):
        return [fix_batch_shape_recursive(item) for item in obj]
    
    return obj

def fix_model_json():
    if not os.path.exists(MODEL_JSON_PATH):
        print(f"Model file not found at {MODEL_JSON_PATH}")
        return False
    
    print(f"Reading model from {MODEL_JSON_PATH}...")
    with open(MODEL_JSON_PATH, 'r') as f:
        model_data = json.load(f)
    
    # Create backup if not exists
    backup_path = MODEL_JSON_PATH + '.keras3.backup'
    if not os.path.exists(backup_path):
        print(f"Creating backup at {backup_path}...")
        with open(backup_path, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    print("Fixing batch_shape -> batchInputShape...")
    model_data = fix_batch_shape_recursive(model_data)
    
    print("Converting Keras 3 inbound_nodes to Keras 2 format...")
    model_data = fix_inbound_nodes_recursive(model_data)
    
    print(f"Writing fixed model to {MODEL_JSON_PATH}...")
    with open(MODEL_JSON_PATH, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print("Model fixed successfully!")
    return True

if __name__ == '__main__':
    fix_model_json()
