import os
from neo4j import GraphDatabase, Driver

def build_neo4j_driver() -> Driver:
    """Create Neo4j driver from environment configuration."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not user or not password:
        raise ValueError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set.")
    
    return GraphDatabase.driver(uri, auth=(user.strip(), password.strip()))

# Add this block at the bottom of your file to test the function!
if __name__ == "__main__":
    try:
        # 1. Build the driver
        my_driver = build_neo4j_driver()
        
        # 2. Force it to connect and test the credentials
        my_driver.verify_connectivity()
        
        print("Success! The driver is built and connected to your Bank-rag database perfectly.")
        
    except Exception as e:
        print(f"Oh no, it failed to connect: {e}")