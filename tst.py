import chromadb

def test_semantic_search():
    # 1. Initialize client and collection
    client = chromadb.Client()
    collection = client.create_collection("my_test")
    documents = ["The feline chased the laser.", "The vehicle has four wheels.", "Data is stored on the hard drive."]
    metadatas = [{"source": "cat.txt"}, {"source": "car.txt"}, {"source": "pc.txt"}]
    ids = ["id1", "id2", "id3"]
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    results = collection.query(query_texts=["How many sick leaves do employees get?"], n_results=2)
    
    print(results)

if __name__ == "__main__":
    test_semantic_search()