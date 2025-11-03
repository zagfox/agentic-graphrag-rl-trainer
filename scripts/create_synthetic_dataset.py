"""
Create synthetic function calling dataset for GRPO training
"""
import json
import random
from datasets import Dataset
from pathlib import Path

def create_synthetic_function_calling_dataset():
    """Create diverse synthetic function calling dataset"""

    # Tool definitions
    tools_registry = {
        "weather": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "include_forecast": {"type": "boolean", "default": False}
                },
                "required": ["location"]
            }
        },
        "search": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5},
                    "language": {"type": "string", "enum": ["en", "es", "fr", "it", "de"]}
                },
                "required": ["query"]
            }
        },
        "calculator": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression"},
                    "precision": {"type": "integer", "default": 2}
                },
                "required": ["expression"]
            }
        },
        "email": {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high"], "default": "normal"}
                },
                "required": ["to", "subject", "body"]
            }
        },
        "database": {
            "name": "query_database",
            "description": "Query a database",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name"},
                    "filters": {"type": "object", "description": "Filter conditions"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["table"]
            }
        }
    }

    # Query templates
    query_templates = [
        # Weather queries
        "What's the weather like in {location}?",
        "Can you check the temperature in {location} in {units}?",
        "Get weather for {location} with forecast",

        # Search queries
        "Search for {topic} on the web",
        "Find information about {topic}",
        "Look up {topic} for me",

        # Calculator queries
        "Calculate {expression}",
        "What is {expression}?",
        "Compute {expression}",

        # Email queries
        "Send email to {email} about {subject}",
        "Email {contact} saying {message}",
        "Contact {email} with subject {subject}",

        # Database queries
        "Query the {table} table",
        "Get records from {table} where {condition}",
        "Search {table} for {criteria}"
    ]

    # Locations, entities for variation
    locations = ["New York", "London", "Tokyo", "Paris", "San Francisco", "Berlin", "Rome", "Madrid", "Sydney", "Toronto"]
    topics = ["machine learning", "weather patterns", "stock market", "climate change", "AI news", "sports results", "political events"]
    expressions = ["2+2", "10*5", "100/4", "sqrt(16)", "2^10", "sin(30)", "log(100)"]
    emails = ["john@example.com", "info@company.com", "support@service.com", "admin@website.org"]
    subjects = ["meeting tomorrow", "project update", "urgent notice", "weekly report"]
    tables = ["users", "products", "orders", "employees", "customers"]

    synthetic_data = []

    for i in range(200):  # Create 200 diverse examples
        # Randomly select 1-3 tools for this example
        available_tools = random.sample(list(tools_registry.values()), random.randint(1, 3))

        # Generate query based on available tools
        tool_names = [tool["name"] for tool in available_tools]

        if "get_weather" in tool_names:
            query = random.choice([
                f"What's the weather like in {random.choice(locations)}?",
                f"Check temperature in {random.choice(locations)} in {random.choice(['celsius', 'fahrenheit'])}."
            ])
            tool_calls = [{
                "name": "get_weather",
                "arguments": {
                    "location": random.choice(locations),
                    "units": random.choice(["celsius", "fahrenheit"]),
                    "include_forecast": random.choice([True, False])
                }
            }]

        elif "search_web" in tool_names:
            query = random.choice([
                f"Search for {random.choice(topics)}",
                f"Find information about {random.choice(topics)}"
            ])
            tool_calls = [{
                "name": "search_web",
                "arguments": {
                    "query": random.choice(topics),
                    "num_results": random.randint(3, 10),
                    "language": random.choice(["en", "es", "fr", "it", "de"])
                }
            }]

        elif "calculate" in tool_names:
            query = random.choice([
                f"Calculate {random.choice(expressions)}",
                f"What is {random.choice(expressions)}?"
            ])
            tool_calls = [{
                "name": "calculate",
                "arguments": {
                    "expression": random.choice(expressions),
                    "precision": random.randint(1, 4)
                }
            }]

        elif "send_email" in tool_names:
            query = random.choice([
                f"Send email to {random.choice(emails)} about {random.choice(subjects)}",
                f"Contact {random.choice(emails)} with subject {random.choice(subjects)}"
            ])
            tool_calls = [{
                "name": "send_email",
                "arguments": {
                    "to": random.choice(emails),
                    "subject": random.choice(subjects),
                    "body": f"Regarding {random.choice(subjects)} - please review",
                    "priority": random.choice(["low", "normal", "high"])
                }
            }]

        else:  # database or others
            query = random.choice([
                f"Query the {random.choice(tables)} table",
                f"Get records from {random.choice(tables)}"
            ])
            tool_calls = [{
                "name": "query_database",
                "arguments": {
                    "table": random.choice(tables),
                    "filters": {"status": "active"},
                    "limit": random.randint(5, 20)
                }
            }]

        synthetic_data.append({
            "tools": available_tools,
            "query": query,
            "tool_calls": tool_calls
        })

    return Dataset.from_list(synthetic_data)

if __name__ == "__main__":
    # Create dataset
    dataset = create_synthetic_function_calling_dataset()

    # Save to data directory
    output_dir = Path("data/synthetic_function_calling")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    with open(output_dir / "train.jsonl", "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")

    # Save dataset object
    dataset.save_to_disk(output_dir / "dataset")

    print(f"✅ Created synthetic dataset with {len(dataset)} examples")
    print(f"📁 Saved to: {output_dir}")

    # Show examples
    print("\n📝 Sample examples:")
    for i in range(3):
        example = dataset[i]
        print(f"\nExample {i+1}:")
        print(f"  Query: {example['query']}")
        print(f"  Tools: {[tool['name'] for tool in example['tools']]}")
        print(f"  Calls: {example['tool_calls']}")