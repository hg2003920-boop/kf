import dotenv, os, re

# Simulate what Agent.load does
dotenv.load_dotenv(r'e:\xm\project\llm_customer_service\ecs_demo\.env')

# Check the env var
key = os.environ.get('DASHSCOPE_API_KEY', '')
print(f"DASHSCOPE_API_KEY from os.environ: {repr(key)} (len={len(key)})")

# Simulate _resolve_env_vars
pattern = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")
test_val = "${DASHSCOPE_API_KEY}"
result = pattern.sub(
    lambda m: os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else ""),
    test_val
)
print(f"Resolved api_key: {repr(result)} (len={len(result)})")

# Also check proxy
print(f"HTTPS_PROXY: {repr(os.environ.get('HTTPS_PROXY', ''))}")
print(f"HTTP_PROXY: {repr(os.environ.get('HTTP_PROXY', ''))}")

# Check if dotenv loaded the env file
print(f"\nAll DASHSCOPE vars: {[(k,v) for k,v in os.environ.items() if 'DASH' in k]}")
#111