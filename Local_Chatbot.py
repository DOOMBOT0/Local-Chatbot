
from ollama import Client
import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


client = chromadb.PersistentClient()
remote_client = Client(host=f"http://localhost:11434")
collection = client.get_or_create_collection(name="articles_demo")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20, separators=["."]
)

with open('counter.txt','r') as f:
    count = int(f.read().strip())


#print("Reading political_news.json and generating embeddings...")
with open("political_news.json", "r", encoding="utf-8") as f:
    json_content = json.load(f)
    for i, article in enumerate(json_content):
        if i<count:
            continue
        count += 1
        content = article["content"]
        sentences = text_splitter.split_text(content)
        #print sentences
        for each_sentence in sentences:
        
            response = remote_client.embed(model="nomic-embed-text", input=f"search_document: {each_sentence}")
            embedding = response["embeddings"][0]

            collection.add(
                ids=[f"article_{i}"],
                embeddings=[embedding],
                documents=[each_sentence],
                metadatas=[{"title": article["title"]}],
            )

#print("Database built successfully!")


with open('counter.txt','w') as f:
    f.write(str(count))

# query = "what are different problems provinces of nepal are facing?"
#query = "are there any predicted hindrance for upcoming election ?"
while True:
    query = input("How may i help you?:  ")
    if query  == 'bye':
        break
    query_embed = remote_client.embed(model="nomic-embed-text", input=f"query: {query}")["embeddings"][0]
    results = collection.query(query_embeddings=[query_embed], n_results=1)
    #print(f"\nQuestion: {query}")
    #print(f'\n Title : {results["metadatas"][0][0]["title"]} \n {results["documents"][0][0]} ')
    context= '\n'.join(results['documents'][0])
    prompt = f"""You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer. If context does not have enough information just say "I don't know"
    context: {context}
    questiion: {query}
    answer: """
    #print(prompt)
    response = remote_client.generate(
            model="qwen2.5:7b-instruct-q4_K_M",
            prompt=prompt,
            options={
                "temperature": 0.1
            }
        )

    answer = response['response']

    print(answer)
    print("===================================")