from chroma_utils import ChromaUtils

def test_query():
    chroma = ChromaUtils(collection_name='obsidian')
    results = chroma.query('Jordan Peterson', n_results=3)
    
    print('\nQuery Results:')
    for i, doc in enumerate(results['documents'][0]):
        print(f'\nResult {i+1}:')
        print(doc)
        print('-' * 50)

if __name__ == '__main__':
    test_query()
