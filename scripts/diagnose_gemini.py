import os
import sys
from dotenv import load_dotenv

print("=" * 50)
print(" Gemini API Diagnostic Tool ")
print("=" * 50)

# 1. Load env
if os.path.exists(".env"):
    load_dotenv()
    print("✓ Loaded .env file successfully.")
else:
    print("✗ .env file not found in current directory.")

google_key = os.environ.get("GOOGLE_API_KEY")
gemini_key = os.environ.get("GEMINI_API_KEY")

key_to_use = google_key or gemini_key

if not key_to_use:
    print("✗ ERROR: No Google API key found in your environment variables.")
    print("  Please make sure you have GOOGLE_API_KEY=your_key in your .env file.")
    sys.exit(1)

key_suffix = key_to_use[-6:] if len(key_to_use) > 12 else ""
print(f"✓ Found API Key: '{key_to_use[:6]}...{key_suffix}' (Length: {len(key_to_use)})")

# 2. Check key format
if not key_to_use.startswith("AIzaSy"):
    print("⚠ WARNING: Your API key does not start with the standard 'AIzaSy' prefix.")
    print("  Google AI Studio developer keys typically start with 'AIzaSy'.")
    print("  If you are using a Vertex AI service account key or a custom proxy, this might be expected,")
    print("  otherwise this key is likely invalid or incomplete.")

# 3. Test list_models using google-generativeai or google-genai
try:
    try:
        import google.generativeai as genai
        legacy = True
    except ImportError:
        from google import genai
        legacy = False
        
    print("✓ Google GenAI SDK package is installed.")
    
    if legacy:
        genai.configure(api_key=key_to_use)
        print("Attempting to fetch list of available models from Google servers (legacy SDK)...")
        models = list(genai.list_models())
        print(f"✓ Success! Connected to Google. Your key has access to {len(models)} models:")
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f"  - {m.name}")
    else:
        client = genai.Client(api_key=key_to_use)
        print("Attempting to fetch list of available models from Google servers (new SDK)...")
        # In the new SDK, models are listed via client.models.list()
        for m in client.models.list():
            print(f"  - {m.name}")
            
except Exception as e:
    print("✗ ERROR: Failed to list models using Google GenAI SDK:")
    print(f"  {e}")
    print("\n  Common fixes:")
    print("  1. Make sure your internet connection is active and not blocked by a firewall.")
    print("  2. Check if the Generative Language API is enabled in your Google Cloud Project console.")
    print("  3. Double check that the key copy-pasted in your .env has no spaces or quotes.")

# 4. Test ChatGoogleGenerativeAI invocation
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("\n✓ langchain-google-genai package is installed.")
    
    # Try gemini-1.5-flash-latest
    print("Testing ChatGoogleGenerativeAI invoke with 'gemini-1.5-flash-latest'...")
    chat = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=key_to_use,
        temperature=0.0
    )
    res = chat.invoke("Hi")
    print(f"✓ Success! Model responded: '{res.content.strip()}'")
except Exception as e:
    print("✗ ERROR: LangChain ChatGoogleGenerativeAI call failed:")
    print(f"  {e}")

print("=" * 50)
