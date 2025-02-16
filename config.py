import toml
from typing import List, Dict, Optional


class Config:
    def __init__(self, config_path: str = "config.toml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return toml.load(f)

    def get_config(self) -> Dict[str, str]:
        return self.config

    @property
    def models(self) -> List[str]:
        return self.config.get("models", [])

    @property
    def aliases(self) -> Dict[str, str]:
        return self.config.get("model_aliases", {})

    @property
    def rag(self) -> Dict[str, str]:
        return self.config.get("rag", {})

    @property
    def chat(self) -> Dict[str, str]:
        return self.config.get("chat", {})

    def get_model_from_alias(self, alias: str) -> str:
        return self.aliases.get(alias, alias)

    # get model alias from model name
    def get_alias_from_model(self, model_name: str) -> str:
        for alias, full_name in self.aliases.items():
            if full_name == model_name:
                return alias
        return model_name

    def save_config(self, config_path: str = None):
        if config_path is None:
            config_path = self.config_path
        with open(config_path, "w") as f:
            toml.dump(self.config, f)

    def get_rag_model(self) -> str:
        """Get current chat model"""
        return self.config.get("rag", {}).get("model_name", "")

    # New methods for chat model configuration
    def get_chat_model(self) -> str:
        """Get current chat model"""
        return self.config.get("chat", {}).get("model_name", "")

    def set_chat_model(self, model_name: str) -> bool:
        """
        Set chat model name. If model_name is an alias, it will be converted to full name.
        Returns True if successful, False if model is not in the allowed models list.
        """
        # Convert alias to full model name if it's an alias
        full_model_name = self.get_model_from_alias(model_name)

        # Verify the model is in the allowed models list
        if full_model_name not in self.models:
            return False

        # Update the config
        if "chat" not in self.config:
            self.config["chat"] = {}
        self.config["chat"]["model_name"] = full_model_name

        # Save the updated config
        self.save_config()
        return True


# Usage example:
if __name__ == "__main__":
    config = Config()
    print("Models:", config.models)
    print("Aliases:", config.aliases)
    print("Current chat model:", config.get_chat_model())

    # Example: change chat model using alias
    new_model = "st"
    if config.set_chat_model(new_model):
        print(f"Successfully changed chat model to: {config.get_chat_model()}")
    else:
        print(f"Failed to change model - '{new_model}' is not a valid model")

    # Example: try to set an invalid model
    invalid_model = "invalid-model"
    if not config.set_chat_model(invalid_model):
        print(f"Failed to change model - '{invalid_model}' is not a valid model")
