import traceback
print('Starting...')
try:
    from sentence_transformers import SentenceTransformer
    print('Imported sentence_transformers')
    model = SentenceTransformer('BAAI/bge-m3')
    print('Loaded BAAI/bge-m3')
except Exception as e:
    traceback.print_exc()
