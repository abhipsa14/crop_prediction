from pymongo import MongoClient
connection_string="mongodb+srv://abhipsasri8183:fccv5v9jXuJIs4W6@cluster1.b94xm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1"
client = MongoClient(connection_string)
database=client['Framer']
collection=database['FramerData']
documents=collection.find()
for document in documents:
    print(document)
print("thank you")