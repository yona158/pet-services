from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import re
from nltk.stem import PorterStemmer



stemmer = PorterStemmer()

load_dotenv()

# Initialize client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Stopwords list for cleaner matching
STOPWORDS = {
    "the", "is", "to", "a", "an", "and", "in", "of", "on", "for", "with",
    "at", "by", "from", "up", "out", "into", "over", "about", "this", "that",
    "as", "it", "be", "are", "was", "were", "can", "we", "you", "i", "my",
    "your", "our", "their", "they", "he", "she", "me", "do", "does", "did",
    "have", "has", "had"
}

def normalize(text: str) -> set:
    words = text.lower().split()
    # return {word for word in words if word not in STOPWORDS}
    # return {word.strip(".,!?") for word in words if word not in STOPWORDS}
    words = re.findall(r'\b\w{3,}\b', text.lower())
    return {stemmer.stem(word) for word in words if word not in STOPWORDS}


def preprocess_service(service):
    combined = f"{service['title']} {service['description']} {' '.join(service.get('keywords', []))}".lower()
    return {
        "service": service,
        "tokens": normalize(combined)
    }

# Preprocess services once at startup
ALL_SERVICES = [preprocess_service(s) for s in json.load(open("services.json", encoding="utf-8"))]

# Matching logic
def get_pet_service_from_query(user_query: str) -> dict:
    user_words = normalize(user_query)
    best_match = None
    best_score = 0

    for entry in ALL_SERVICES:
        score = len(user_words & entry["tokens"])
        if score > best_score:
            best_score = score
            best_match = entry["service"]

    if best_match and best_score > 0:
        return {
            "title": best_match["title"],
            "description": best_match["description"]
        }

    return {
        "title": "No suitable service found",
        "description": "Sorry, we couldn't match your request to any available service."
    }

# Tool schema declaration
pet_services = {
    "name": "get_pet_service_from_query",
    "description": (
        "Use this tool when a user describes a pet-related need or situation that might require a service. "
        "This includes travel, illness, behavior problems, grooming, sitting, or general care. "
        "Always use this tool when the user asks for help without naming a specific service."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_query": {
                "type": "string",
                "description": "A short user message describing their pet's issue or need, like 'my dog is aggressive'."
            },
        },
        "required": ["user_query"],
    },
}

tools = types.Tool(function_declarations=[pet_services])
config = types.GenerateContentConfig(tools=[tools])

# Main loop
def chat():
    contents = []
    print("Pet Assistant — type 'exit' to quit.\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Add user input to chat
        contents.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        try:
            # Ask Gemini
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                config=config,
                contents=contents,
            )

            part = response.candidates[0].content.parts[0]

            if hasattr(part, "function_call") and part.function_call:
                tool_call = part.function_call

                result = get_pet_service_from_query(**tool_call.args)

                # function_response_part = types.Part.from_function_response(
                #     name=tool_call.name,
                #     response={"result": result},
                # )
                
                prompt = (
                    f"The user's message was: '{tool_call.args['user_query']}'.\n"
                    f"The best-matching service is:\n\n"
                    f"Title: {result['title']}\n"
                    f"Description: {result['description']}\n\n"
                    f"Please explain this service naturally to the user."
                )

    

                contents.append(response.candidates[0].content)
                contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

                final_response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    config=config,
                    contents=contents,
                )

                print("\nGemini:", final_response.text)
                contents.append(final_response.candidates[0].content)

            else:  
                print("No function call made. Matching locally...")

                result = get_pet_service_from_query(user_input)

                prompt = (
                    f"The user's message was: '{user_input}'.\n"
                    f"The best-matching service is:\n\n"
                    f"Title: {result['title']}\n"
                    f"Description: {result['description']}\n\n"
                    f"Please explain this service naturally to the user."
                )

                contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

                friendly_response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    config=config,
                    contents=contents,
                )

                print("\nGemini:", friendly_response.text)
                contents.append(friendly_response.candidates[0].content)

        except Exception as e:
            print("Error:", str(e))

chat()
