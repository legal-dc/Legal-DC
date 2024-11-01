import elasticsearch

class retriever_bm25:

    def __init__(self,url,index_name) -> None:
        self.url=url
        self.index_name=index_name


    def retrieve(self,query):
        
        elasticsearch_url = self.url
        client=elasticsearch.Elasticsearch(elasticsearch_url)

        index_name=self.index_name
        query_dict = {"query": {"match": {"content": query}}}

        res = client.search(index=index_name, body=query_dict)

        docs = []
        for r in res["hits"]["hits"]:
            # docs.append(Document(page_content=r["_source"]["content"]))
            docs.append(r["_source"]["content"])

        return docs

class test1:
     pass

def main():
        elasticsearch_url = "http://10.122.231.38:9200/"
        index_name="langchain-index-4"   
        query='foo'
        bm25_retriever =retriever_bm25(elasticsearch_url,index_name)
 
        print(bm25_retriever.retrieve(query))
    
if __name__=='__main__':
    main()
    
