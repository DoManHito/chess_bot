import json

with open('chess_move_classifier_eda.ipynb', 'r') as f:
    data = json.load(f)

for i, cell in enumerate(data['cells']):
    cell_type = cell.get('cell_type', '')
    source = cell.get('source', [])
    if isinstance(source, list):
        source_text = ''.join(source)
    else:
        source_text = source
    
    if cell_type == 'code':
        print(f'Cell {i}:')
        print(source_text[:200])
        print('---')
