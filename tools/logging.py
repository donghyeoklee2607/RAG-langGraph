import os


def langsmith(project_name=None, set_enable=True):

    if set_enable:
        langchain_key = os.environ.get("LANGCHAIN_API_KEY", "")
        langsmith_key = os.environ.get("LANGSMITH_API_KEY", "")

        if len(langchain_key.strip()) >= len(langsmith_key.strip()):
            result = langchain_key
        else:
            result = langsmith_key

        if result.strip() == "":
            print(
                "LangChain/LangSmith API Key가 is not set up in .env file."
            )
            return

        os.environ["LANGSMITH_ENDPOINT"] = (
            "https://api.smith.langchain.com" 
        )
        os.environ["LANGSMITH_TRACING"] = "true" 
        os.environ["LANGSMITH_PROJECT"] = project_name 
        print(f"LangSmith is tracing....\n[Project Name]\n{project_name}")
    else:
        os.environ["LANGSMITH_TRACING"] = "false" 
        print("LangSmith is not tracing...")


def env_variable(key, value):
    os.environ[key] = value