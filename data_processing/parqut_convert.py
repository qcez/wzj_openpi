import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# Assuming your input Parquet file is named 'input.parquet'
# Replace with your actual file path
input_file = '/home/congcong/.cache/huggingface/lerobot/folding_clothes/data/chunk-000/file-002.parquet'
output_file = '/home/congcong/.cache/huggingface/lerobot/folding_clothes/data/chunk-000/file-002.parquet'

# Read the Parquet file using pyarrow
table = pq.read_table(input_file)

# Function to safely extract the first element from a list (assuming lists have length 1 for scalar fields)
def extract_scalar_from_list(array):
    # Convert pyarrow array to numpy for easier manipulation
    np_array = array.to_numpy()
    extracted = np.full(len(np_array), None, dtype=object)  # 修正：使用 None 而非 pa.NULL_VALUE
    for i, item in enumerate(np_array):
        if item is not None and len(item) > 0:
            extracted[i] = item[0]
        else:
            extracted[i] = None  # Use None for null
    # Convert back to pyarrow array with appropriate type (will cast later)
    return pa.array(extracted)

# Define scalar fields that need extraction and type casting
scalar_fields = ['time_stamp', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']

# Process each scalar field: extract from list and cast to proper type
for field in scalar_fields:
    if field in table.column_names:
        # Extract the scalar value
        extracted_array = extract_scalar_from_list(table[field])
        
        # Cast to appropriate type
        if 'time' in field or 'stamp' in field:
            # Float type for timestamps
            casted_array = pa.compute.cast(extracted_array, pa.float32())
        else:
            # Int type for indices (pyarrow int64 handles nulls)
            casted_array = pa.compute.cast(extracted_array, pa.int64())
        
        # Replace the column in the table
        table = table.set_column(table.schema.get_field_index(field), field, casted_array)

# Ensure observation.state and action are lists of float32
if 'observation.state' in table.column_names:
    # Cast the list elements to float32 if needed (pyarrow preserves list structure)
    casted_list = pa.compute.cast(table['observation.state'], pa.list_(pa.float32()))
    table = table.set_column(table.schema.get_field_index('observation.state'), 'observation.state', casted_list)

if 'action' in table.column_names:
    casted_list = pa.compute.cast(table['action'], pa.list_(pa.float32()))
    table = table.set_column(table.schema.get_field_index('action'), 'action', casted_list)

# Write the updated table to a new Parquet file
pq.write_table(table, output_file)

print(f"Conversion complete. New schema applied in {output_file}")
# Optional: Print schema for verification
print(table.schema)