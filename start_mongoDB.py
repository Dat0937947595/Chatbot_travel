from pymongo import MongoClient

def start_mongo():
    client = MongoClient('mongodb+srv://minhdat:1111@travelchatbot.izjgp.mongodb.net/')
    db = client['travel_db']
    return db