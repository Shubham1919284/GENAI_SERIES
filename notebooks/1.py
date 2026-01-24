from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# Load environment variables (try multiple paths)
env_loaded = load_dotenv('.env') or load_dotenv('notebooks/.env')

print("Environment loaded:", env_loaded)
print("OPENROUTER_SONET_API_KEY loaded:", bool(os.getenv('OPENROUTER_SONET_API_KEY')))

# For OpenRouter, we need to set OPENAI_API_KEY to the OpenRouter key
openrouter_key = os.getenv('OPENROUTER_SONET_API_KEY')  # Note: You have "SONET" not "SONNET" in your .env
if openrouter_key:
    os.environ['OPENAI_API_KEY'] = openrouter_key
    print("✅ API key set successfully")
else:
    print("❌ OPENROUTER_SONET_API_KEY not found")
    exit(1)

# Claude 3.5 Sonnet via OpenRouter (Claude 4.5 Sonnet doesn't exist yet)
print("\n🚨 Note: Claude 4.5 Sonnet doesn't exist. Using Claude 3.5 Sonnet instead.")
print("⚠️  You have limited credits. Trying with fewer tokens...")

claude_llm = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    max_tokens=500  # Reduced from 1000 to fit your credits
)

try:
    # Test the model
    response = claude_llm.invoke("Hello! Can you introduce yourself?")
    print("\nClaude 3.5 Sonnet Response:")
    print(response.content)
except Exception as e:
    print(f"\n❌ Claude 3.5 Sonnet failed: {e}")
    print("\n🔄 Trying cheaper Claude 3 Haiku model...")

    # Try cheaper model
    cheap_claude = ChatOpenAI(
        model="anthropic/claude-3-haiku",
        base_url="https://openrouter.ai/api/v1",
        max_tokens=300
    )

    try:
        response = cheap_claude.invoke("Hello! Can you introduce yourself?")
        print("\nClaude 3 Haiku Response:")
        print(response.content)
    except Exception as e2:
        print(f"❌ Even Claude Haiku failed: {e2}")
        print("\n💡 To get more credits, visit: https://openrouter.ai/settings/credits")